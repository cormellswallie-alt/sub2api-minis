package service

// Signed thinking passback for the Chat Completions -> Anthropic bridge.
//
// Chat Completions has no field for thinking signatures: clients only echo the
// visible summary in `reasoning_content`. Anthropic accepts prior thinking only
// as the original signed block, and the signature (not the summary) is what
// carries the full reasoning back to the model.
//
// So the bridge remembers every signed thinking block it streams out (Redis,
// keyed by the tool_use ids of that response and by a whitespace-insensitive
// hash of the emitted reasoning text). On the next request, each assistant
// turn whose tool calls or reasoning text match gets its original signed
// blocks re-inserted at the front. Turns with no cache entry fall back to the
// summary as plain text (SUB2API_CHAT_THINKING_FALLBACK=text, default) or are
// dropped (=drop). If the upstream still rejects a signature, the request is
// retried once without any thinking blocks.
//
// Disable entirely with SUB2API_CHAT_SIGNED_THINKING=false.

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

const (
	chatSigKeyPrefix   = "ccsig:"
	chatSigTTL         = 14 * 24 * time.Hour
	chatSigTrackerKey  = "minis_chat_sig_tracker"
	chatSigMaxLookups  = 400
	chatSigLookupLimit = 3 * time.Second
)

func chatEnvBool(name string, def bool) bool {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

func chatSignedThinkingEnabled() bool { return chatEnvBool("SUB2API_CHAT_SIGNED_THINKING", true) }

// chatThinkingTextFallback: keep the summary as text when no signed block exists.
func chatThinkingTextFallback() bool {
	return !strings.EqualFold(strings.TrimSpace(os.Getenv("SUB2API_CHAT_THINKING_FALLBACK")), "drop")
}

// chatSigStore is the subset of GatewayCache used here (keeps tests simple).
type chatSigStore interface {
	SetReasoningContent(ctx context.Context, itemID string, content string, ttl time.Duration) error
	GetReasoningContent(ctx context.Context, itemID string) (string, error)
}

// chatSigBlock is an Anthropic thinking or redacted_thinking block, verbatim.
type chatSigBlock struct {
	Type      string `json:"type"`
	Thinking  string `json:"thinking,omitempty"`
	Signature string `json:"signature,omitempty"`
	Data      string `json:"data,omitempty"`
}

type chatSigEntry struct {
	Model   string         `json:"model"`
	Blocks  []chatSigBlock `json:"blocks"`
	ToolIDs []string       `json:"tool_ids,omitempty"`
}

func chatSigNormalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if !unicode.IsSpace(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func chatSigTextKey(text string) string {
	n := chatSigNormalize(text)
	if n == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(n))
	return chatSigKeyPrefix + "t:" + hex.EncodeToString(sum[:16])
}

// chatSigToolID strips bridge prefixes so client ids and Anthropic ids compare equal.
func chatSigToolID(id string) string {
	id = strings.TrimSpace(id)
	if after, ok := strings.CutPrefix(id, "fc_"); ok {
		id = after
	}
	return id
}

func chatSigToolKey(id string) string {
	id = chatSigToolID(id)
	if id == "" {
		return ""
	}
	return chatSigKeyPrefix + "c:" + id
}

// ---------------------------------------------------------------------------
// Capture: observe the raw Anthropic stream of one response.

type chatSigTracker struct {
	model   string // client-facing model; entries only replay for the same model
	blocks  map[int]*chatSigBlock
	order   []int
	toolIDs []string
	stopped bool
	saved   bool
}

func newChatSigTracker(model string) *chatSigTracker {
	return &chatSigTracker{model: strings.ToLower(strings.TrimSpace(model)), blocks: map[int]*chatSigBlock{}}
}

func (t *chatSigTracker) observe(ev *apicompat.AnthropicStreamEvent) {
	if t == nil || ev == nil {
		return
	}
	switch ev.Type {
	case "content_block_start":
		if ev.Index == nil || ev.ContentBlock == nil {
			return
		}
		cb := ev.ContentBlock
		switch cb.Type {
		case "thinking", "redacted_thinking":
			if _, ok := t.blocks[*ev.Index]; !ok {
				t.order = append(t.order, *ev.Index)
			}
			t.blocks[*ev.Index] = &chatSigBlock{Type: cb.Type, Thinking: cb.Thinking, Signature: cb.Signature, Data: cb.Data}
		case "tool_use":
			if cb.ID != "" {
				t.toolIDs = append(t.toolIDs, cb.ID)
			}
		}
	case "content_block_delta":
		if ev.Index == nil || ev.Delta == nil {
			return
		}
		b := t.blocks[*ev.Index]
		if b == nil {
			return
		}
		switch ev.Delta.Type {
		case "thinking_delta":
			b.Thinking += ev.Delta.Thinking
		case "signature_delta":
			b.Signature += ev.Delta.Signature
		}
	case "message_stop":
		t.stopped = true
	}
}

// entry returns the signed blocks of a completed response, or nil.
func (t *chatSigTracker) entry() (*chatSigEntry, string) {
	if t == nil || !t.stopped || len(t.order) == 0 {
		return nil, ""
	}
	e := &chatSigEntry{Model: t.model, ToolIDs: t.toolIDs}
	var text strings.Builder
	for _, idx := range t.order {
		b := t.blocks[idx]
		switch {
		case b.Type == "thinking" && b.Signature != "":
			e.Blocks = append(e.Blocks, *b)
			text.WriteString(b.Thinking)
		case b.Type == "redacted_thinking" && b.Data != "":
			e.Blocks = append(e.Blocks, chatSigBlock{Type: b.Type, Data: b.Data})
		}
	}
	if len(e.Blocks) == 0 {
		return nil, ""
	}
	return e, text.String()
}

func (t *chatSigTracker) save(ctx context.Context, store chatSigStore) int {
	if t == nil || t.saved || store == nil {
		return 0
	}
	e, text := t.entry()
	if e == nil {
		return 0
	}
	t.saved = true
	raw, err := json.Marshal(e)
	if err != nil {
		return 0
	}
	keys := []string{chatSigTextKey(text)}
	for _, id := range e.ToolIDs {
		keys = append(keys, chatSigToolKey(id))
	}
	n := 0
	for _, k := range keys {
		if k == "" {
			continue
		}
		if err := store.SetReasoningContent(ctx, k, string(raw), chatSigTTL); err == nil {
			n++
		}
	}
	return n
}

// attachChatSigTracker registers a tracker for this request (nil when disabled).
func (s *GatewayService) attachChatSigTracker(c *gin.Context, model string) {
	if c == nil || s == nil || s.cache == nil || !chatSignedThinkingEnabled() {
		return
	}
	c.Set(chatSigTrackerKey, newChatSigTracker(model))
}

// observeChatSig feeds one upstream event; saves on message_stop.
func (s *GatewayService) observeChatSig(c *gin.Context, ev *apicompat.AnthropicStreamEvent) {
	if c == nil || s == nil || s.cache == nil {
		return
	}
	v, ok := c.Get(chatSigTrackerKey)
	if !ok {
		return
	}
	t, _ := v.(*chatSigTracker)
	if t == nil {
		return
	}
	t.observe(ev)
	if ev != nil && ev.Type == "message_stop" {
		ctx, cancel := context.WithTimeout(context.Background(), chatSigLookupLimit)
		defer cancel()
		if n := t.save(ctx, s.cache); n > 0 {
			logger.L().Debug("forward_as_cc: stored signed thinking", zap.Int("keys", n))
		}
	}
}

// ---------------------------------------------------------------------------
// Replay: find signed blocks for assistant history turns.

// chatSigRef is the lookup result for one assistant message (by message index).
type chatSigRef struct {
	msgIndex  int
	entry     *chatSigEntry
	reasoning string
}

func chatSigReasoning(m *apicompat.ChatMessage) string {
	if m.ReasoningContent != "" {
		return m.ReasoningContent
	}
	return m.Reasoning
}

func chatSigGet(ctx context.Context, store chatSigStore, key, model string) *chatSigEntry {
	if key == "" {
		return nil
	}
	raw, err := store.GetReasoningContent(ctx, key)
	if err != nil || raw == "" {
		return nil
	}
	var e chatSigEntry
	if json.Unmarshal([]byte(raw), &e) != nil || len(e.Blocks) == 0 {
		return nil
	}
	if e.Model != "" && model != "" && e.Model != strings.ToLower(strings.TrimSpace(model)) {
		return nil
	}
	return &e
}

// lookupChatSignedThinking runs before history sanitising (needs reasoning_content).
func lookupChatSignedThinking(ctx context.Context, store chatSigStore, req *apicompat.ChatCompletionsRequest) []chatSigRef {
	if store == nil || req == nil {
		return nil
	}
	ctx, cancel := context.WithTimeout(ctx, chatSigLookupLimit)
	defer cancel()
	var refs []chatSigRef
	lookups := 0
	// Newest turns matter most: walk backwards, then restore original order.
	for i := len(req.Messages) - 1; i >= 0; i-- {
		m := &req.Messages[i]
		if m.Role != "assistant" {
			continue
		}
		reasoning := chatSigReasoning(m)
		if reasoning == "" && len(m.ToolCalls) == 0 {
			continue
		}
		var e *chatSigEntry
		if lookups < chatSigMaxLookups {
			for _, tc := range m.ToolCalls {
				lookups++
				if e = chatSigGet(ctx, store, chatSigToolKey(tc.ID), req.Model); e != nil {
					break
				}
			}
			if e == nil && reasoning != "" {
				lookups++
				e = chatSigGet(ctx, store, chatSigTextKey(reasoning), req.Model)
			}
		}
		if e == nil && reasoning == "" {
			continue
		}
		refs = append(refs, chatSigRef{msgIndex: i, entry: e, reasoning: reasoning})
	}
	for l, r := 0, len(refs)-1; l < r; l, r = l+1, r-1 {
		refs[l], refs[r] = refs[r], refs[l]
	}
	return refs
}

// applyChatThinkingTextFallback re-adds the summary text for turns with no
// signed block (runs after sanitising). Returns the number of turns.
func applyChatThinkingTextFallback(req *apicompat.ChatCompletionsRequest, refs []chatSigRef) int {
	if req == nil || !chatThinkingTextFallback() {
		return 0
	}
	n := 0
	for _, r := range refs {
		if r.entry != nil || r.reasoning == "" || r.msgIndex >= len(req.Messages) {
			continue
		}
		req.Messages[r.msgIndex].ReasoningContent = r.reasoning
		n++
	}
	return n
}

func chatSigPlainText(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var b strings.Builder
	for _, p := range gjson.ParseBytes(raw).Array() {
		switch p.Get("type").String() {
		case "text", "output_text":
			b.WriteString(p.Get("text").String())
		}
	}
	return b.String()
}

// restoreChatSignedThinking prepends cached signed blocks to the matching
// Anthropic assistant messages. Returns the number of turns restored.
func restoreChatSignedThinking(areq *apicompat.AnthropicRequest, creq *apicompat.ChatCompletionsRequest, refs []chatSigRef) int {
	if areq == nil || creq == nil {
		return 0
	}
	// Signed blocks are only valid while thinking is on; otherwise replay the
	// summary as text (same as the text fallback) so nothing is lost.
	asText := areq.Thinking != nil && strings.EqualFold(areq.Thinking.Type, "disabled")
	if asText && !chatThinkingTextFallback() {
		return 0
	}
	hits := make([]chatSigRef, 0, len(refs))
	for _, r := range refs {
		if r.entry != nil {
			hits = append(hits, r)
		}
	}
	if len(hits) == 0 {
		return 0
	}
	restored, cursor := 0, 0
	for mi := range areq.Messages {
		if cursor >= len(hits) {
			break
		}
		am := &areq.Messages[mi]
		if am.Role != "assistant" {
			continue
		}
		var blocks []json.RawMessage
		var str string
		if json.Unmarshal(am.Content, &str) == nil {
			tb, _ := json.Marshal(map[string]string{"type": "text", "text": str})
			blocks = []json.RawMessage{tb}
		} else if json.Unmarshal(am.Content, &blocks) != nil {
			continue
		}
		if len(blocks) > 0 {
			if t := gjson.GetBytes(blocks[0], "type").String(); t == "thinking" || t == "redacted_thinking" {
				continue
			}
		}
		ids := map[string]bool{}
		var text strings.Builder
		for _, b := range blocks {
			switch gjson.GetBytes(b, "type").String() {
			case "tool_use":
				ids[chatSigToolID(gjson.GetBytes(b, "id").String())] = true
			case "text":
				text.WriteString(gjson.GetBytes(b, "text").String())
			}
		}
		normText := chatSigNormalize(text.String())
		for k := cursor; k < len(hits); k++ {
			h := hits[k]
			match := false
			for _, id := range h.entry.ToolIDs {
				if ids[chatSigToolID(id)] {
					match = true
					break
				}
			}
			if !match && len(ids) == 0 && h.msgIndex < len(creq.Messages) {
				cm := &creq.Messages[h.msgIndex]
				match = len(cm.ToolCalls) == 0 && chatSigNormalize(chatSigPlainText(cm.Content)) == normText
			}
			if !match {
				continue
			}
			out := make([]json.RawMessage, 0, len(h.entry.Blocks)+len(blocks))
			if asText {
				if h.reasoning != "" {
					tb, _ := json.Marshal(map[string]string{"type": "text", "text": "<thinking>" + h.reasoning + "</thinking>"})
					out = append(out, tb)
				}
			} else {
				for _, sb := range h.entry.Blocks {
					raw, err := json.Marshal(sb)
					if err != nil {
						continue
					}
					out = append(out, raw)
				}
			}
			if len(out) == 0 {
				cursor = k + 1
				break
			}
			out = append(out, blocks...)
			if raw, err := json.Marshal(out); err == nil {
				am.Content = raw
				restored++
			}
			cursor = k + 1
			break
		}
	}
	return restored
}

// stripAnthropicThinkingBlocks removes every thinking/redacted_thinking block
// from message history (retry path after a signature rejection).
func stripAnthropicThinkingBlocks(body []byte) ([]byte, int) {
	removed := 0
	msgs := gjson.GetBytes(body, "messages")
	if !msgs.IsArray() {
		return body, 0
	}
	out := body
	for i, m := range msgs.Array() {
		content := m.Get("content")
		if !content.IsArray() {
			continue
		}
		kept := make([]json.RawMessage, 0, len(content.Array()))
		changed := false
		for _, b := range content.Array() {
			if t := b.Get("type").String(); t == "thinking" || t == "redacted_thinking" {
				changed = true
				removed++
				continue
			}
			kept = append(kept, json.RawMessage(b.Raw))
		}
		if !changed {
			continue
		}
		if len(kept) == 0 {
			kept = append(kept, json.RawMessage(`{"type":"text","text":"(omitted)"}`))
		}
		raw, err := json.Marshal(kept)
		if err != nil {
			continue
		}
		if next, err := sjson.SetRawBytes(out, "messages."+strconv.Itoa(i)+".content", raw); err == nil {
			out = next
		}
	}
	return out, removed
}

func chatSigRefStats(refs []chatSigRef) (hits, misses int) {
	for _, r := range refs {
		if r.entry != nil {
			hits++
		} else {
			misses++
		}
	}
	return
}
