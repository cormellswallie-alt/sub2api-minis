package service

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/tidwall/gjson"
)

type mapSigStore map[string]string

func (m mapSigStore) SetReasoningContent(_ context.Context, k, v string, _ time.Duration) error {
	m[k] = v
	return nil
}

func (m mapSigStore) GetReasoningContent(_ context.Context, k string) (string, error) {
	if v, ok := m[k]; ok {
		return v, nil
	}
	return "", ErrReasoningContentNotFound
}

func sigIdx(i int) *int { return &i }

// feed simulates one upstream response: thinking (2 deltas + signature), text, tool_use.
func sigFeed(t *chatSigTracker, thinking1, thinking2, sig, toolID string) {
	evs := []apicompat.AnthropicStreamEvent{
		{Type: "message_start"},
		{Type: "content_block_start", Index: sigIdx(0), ContentBlock: &apicompat.AnthropicContentBlock{Type: "thinking"}},
		{Type: "content_block_delta", Index: sigIdx(0), Delta: &apicompat.AnthropicDelta{Type: "thinking_delta", Thinking: thinking1}},
		{Type: "content_block_delta", Index: sigIdx(0), Delta: &apicompat.AnthropicDelta{Type: "thinking_delta", Thinking: thinking2}},
		{Type: "content_block_delta", Index: sigIdx(0), Delta: &apicompat.AnthropicDelta{Type: "signature_delta", Signature: sig}},
		{Type: "content_block_start", Index: sigIdx(1), ContentBlock: &apicompat.AnthropicContentBlock{Type: "text"}},
	}
	if toolID != "" {
		evs = append(evs, apicompat.AnthropicStreamEvent{Type: "content_block_start", Index: sigIdx(2),
			ContentBlock: &apicompat.AnthropicContentBlock{Type: "tool_use", ID: toolID, Name: "file_write"}})
	}
	evs = append(evs, apicompat.AnthropicStreamEvent{Type: "message_stop"})
	for i := range evs {
		t.observe(&evs[i])
	}
}

func TestSignedThinkingCaptureAndLookup(t *testing.T) {
	store := mapSigStore{}
	tr := newChatSigTracker("claude-opus-5-5")
	sigFeed(tr, "first half, ", "second half", "SIG-A", "toolu_01")
	if n := tr.save(context.Background(), store); n != 2 {
		t.Fatalf("saved keys=%d store=%v", n, store)
	}
	// Second save is a no-op.
	if n := tr.save(context.Background(), store); n != 0 {
		t.Fatalf("double save: %d", n)
	}

	req := &apicompat.ChatCompletionsRequest{Model: "claude-opus-5-5", Messages: []apicompat.ChatMessage{
		{Role: "user", Content: json.RawMessage(`"q"`)},
		// Matched by tool id (client sees the id unchanged).
		{Role: "assistant", ReasoningContent: "edited text", ToolCalls: []apicompat.ChatToolCall{{ID: "toolu_01", Type: "function",
			Function: apicompat.ChatFunctionCall{Name: "file_write", Arguments: "{}"}}}},
		{Role: "tool", ToolCallID: "toolu_01", Content: json.RawMessage(`"ok"`)},
		// Matched by text hash despite whitespace changes.
		{Role: "assistant", ReasoningContent: "  first half,\nsecond   half ", Content: json.RawMessage(`"done"`)},
		// Miss: no cache entry.
		{Role: "assistant", ReasoningContent: "unknown", Content: json.RawMessage(`"x"`)},
	}}
	refs := lookupChatSignedThinking(context.Background(), store, req)
	if len(refs) != 3 {
		t.Fatalf("refs=%d", len(refs))
	}
	if refs[0].msgIndex != 1 || refs[0].entry == nil || refs[0].entry.Blocks[0].Signature != "SIG-A" {
		t.Errorf("tool-id ref: %+v", refs[0])
	}
	if refs[0].entry.Blocks[0].Thinking != "first half, second half" {
		t.Errorf("thinking text: %q", refs[0].entry.Blocks[0].Thinking)
	}
	if refs[1].msgIndex != 3 || refs[1].entry == nil {
		t.Errorf("text-hash ref: %+v", refs[1])
	}
	if refs[2].msgIndex != 4 || refs[2].entry != nil || refs[2].reasoning != "unknown" {
		t.Errorf("miss ref: %+v", refs[2])
	}
	if h, m := chatSigRefStats(refs); h != 2 || m != 1 {
		t.Errorf("stats %d/%d", h, m)
	}

	// Different model: never replay a foreign signature.
	req.Model = "claude-sonnet-5-5"
	for _, r := range lookupChatSignedThinking(context.Background(), store, req) {
		if r.entry != nil {
			t.Errorf("cross-model hit at %d", r.msgIndex)
		}
	}
}

func TestSignedThinkingRedactedAndUnsigned(t *testing.T) {
	tr := newChatSigTracker("m")
	evs := []apicompat.AnthropicStreamEvent{
		{Type: "content_block_start", Index: sigIdx(0), ContentBlock: &apicompat.AnthropicContentBlock{Type: "thinking"}},
		{Type: "content_block_delta", Index: sigIdx(0), Delta: &apicompat.AnthropicDelta{Type: "thinking_delta", Thinking: "no sig"}},
		{Type: "content_block_start", Index: sigIdx(1), ContentBlock: &apicompat.AnthropicContentBlock{Type: "redacted_thinking", Data: "ENC"}},
		{Type: "message_stop"},
	}
	for i := range evs {
		tr.observe(&evs[i])
	}
	e, text := tr.entry()
	if e == nil || len(e.Blocks) != 1 || e.Blocks[0].Type != "redacted_thinking" || e.Blocks[0].Data != "ENC" || text != "" {
		t.Fatalf("entry=%+v text=%q", e, text)
	}
	// Incomplete stream (no message_stop) is never stored.
	tr2 := newChatSigTracker("m")
	sigFeed(tr2, "a", "b", "S", "")
	tr2.stopped = false
	if e, _ := tr2.entry(); e != nil {
		t.Error("incomplete response stored")
	}
}

func sigAnthropicReq(thinking string) *apicompat.AnthropicRequest {
	r := &apicompat.AnthropicRequest{Messages: []apicompat.AnthropicMessage{
		{Role: "user", Content: json.RawMessage(`"q"`)},
		{Role: "assistant", Content: json.RawMessage(`[{"type":"tool_use","id":"toolu_01","name":"file_write","input":{}}]`)},
		{Role: "user", Content: json.RawMessage(`[{"type":"tool_result","tool_use_id":"toolu_01","content":"ok"}]`)},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "user", Content: json.RawMessage(`"next"`)},
	}}
	if thinking != "" {
		r.Thinking = &apicompat.AnthropicThinking{Type: thinking}
	}
	return r
}

func sigChatReq() *apicompat.ChatCompletionsRequest {
	return &apicompat.ChatCompletionsRequest{Model: "claude-opus-5-5", Messages: []apicompat.ChatMessage{
		{Role: "user", Content: json.RawMessage(`"q"`)},
		{Role: "assistant", ToolCalls: []apicompat.ChatToolCall{{ID: "toolu_01"}}},
		{Role: "tool", ToolCallID: "toolu_01"},
		{Role: "assistant", Content: json.RawMessage(`"done"`)},
		{Role: "user", Content: json.RawMessage(`"next"`)},
	}}
}

func sigRefs() []chatSigRef {
	return []chatSigRef{
		{msgIndex: 1, reasoning: "plan A", entry: &chatSigEntry{ToolIDs: []string{"toolu_01"},
			Blocks: []chatSigBlock{{Type: "thinking", Thinking: "plan A", Signature: "SIG-1"}}}},
		{msgIndex: 3, reasoning: "plan B", entry: &chatSigEntry{
			Blocks: []chatSigBlock{{Type: "thinking", Thinking: "plan B", Signature: "SIG-2"}, {Type: "redacted_thinking", Data: "D"}}}},
	}
}

func TestSignedThinkingRestore(t *testing.T) {
	areq := sigAnthropicReq("adaptive")
	if n := restoreChatSignedThinking(areq, sigChatReq(), sigRefs()); n != 2 {
		t.Fatalf("restored=%d", n)
	}
	m1 := gjson.ParseBytes(areq.Messages[1].Content).Array()
	if len(m1) != 2 || m1[0].Get("type").String() != "thinking" || m1[0].Get("signature").String() != "SIG-1" ||
		m1[1].Get("type").String() != "tool_use" {
		t.Errorf("tool turn: %s", areq.Messages[1].Content)
	}
	m3 := gjson.ParseBytes(areq.Messages[3].Content).Array()
	if len(m3) != 3 || m3[0].Get("signature").String() != "SIG-2" || m3[1].Get("data").String() != "D" ||
		m3[2].Get("text").String() != "done" {
		t.Errorf("text turn: %s", areq.Messages[3].Content)
	}
	// Idempotent: a message that already starts with thinking is left alone.
	if n := restoreChatSignedThinking(areq, sigChatReq(), sigRefs()); n != 0 {
		t.Errorf("second restore=%d", n)
	}
}

func TestSignedThinkingRestoreWhenThinkingDisabled(t *testing.T) {
	areq := sigAnthropicReq("disabled")
	if n := restoreChatSignedThinking(areq, sigChatReq(), sigRefs()); n != 2 {
		t.Fatalf("restored=%d", n)
	}
	if strings.Contains(string(areq.Messages[1].Content), "signature") {
		t.Errorf("signed block sent with thinking disabled: %s", areq.Messages[1].Content)
	}
	if gjson.GetBytes(areq.Messages[1].Content, "0.text").String() != "<thinking>plan A</thinking>" {
		t.Errorf("text replay: %s", areq.Messages[1].Content)
	}
	t.Setenv("SUB2API_CHAT_THINKING_FALLBACK", "drop")
	if n := restoreChatSignedThinking(sigAnthropicReq("disabled"), sigChatReq(), sigRefs()); n != 0 {
		t.Errorf("drop mode restored %d", n)
	}
}

func TestSignedThinkingTextFallback(t *testing.T) {
	req := sigChatReq()
	refs := []chatSigRef{{msgIndex: 3, reasoning: "kept summary"}, {msgIndex: 1, reasoning: "hit", entry: &chatSigEntry{}}}
	if n := applyChatThinkingTextFallback(req, refs); n != 1 || req.Messages[3].ReasoningContent != "kept summary" ||
		req.Messages[1].ReasoningContent != "" {
		t.Errorf("fallback n=%d msgs=%+v", n, req.Messages)
	}
	t.Setenv("SUB2API_CHAT_THINKING_FALLBACK", "drop")
	if n := applyChatThinkingTextFallback(sigChatReq(), refs); n != 0 {
		t.Errorf("drop mode fallback=%d", n)
	}
}

func TestStripAnthropicThinkingBlocks(t *testing.T) {
	body := []byte(`{"model":"m","messages":[{"role":"user","content":"q"},` +
		`{"role":"assistant","content":[{"type":"thinking","thinking":"t","signature":"s"},{"type":"text","text":"a"}]},` +
		`{"role":"assistant","content":[{"type":"redacted_thinking","data":"d"}]},{"role":"user","content":"n"}]}`)
	out, n := stripAnthropicThinkingBlocks(body)
	if n != 2 {
		t.Fatalf("removed=%d", n)
	}
	if strings.Contains(string(out), "thinking") {
		t.Errorf("thinking left: %s", out)
	}
	if gjson.GetBytes(out, "messages.1.content.0.text").String() != "a" ||
		gjson.GetBytes(out, "messages.2.content.0.text").String() != "(omitted)" ||
		gjson.GetBytes(out, "messages.0.content").String() != "q" {
		t.Errorf("structure: %s", out)
	}
}
