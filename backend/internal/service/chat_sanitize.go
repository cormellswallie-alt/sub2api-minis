package service

// Local fixes for the Chat Completions -> Anthropic bridge:
//
//  1. History hygiene: clients echo `reasoning_content` on assistant turns and
//     the upstream converter inlines it as literal tagged text in the assistant
//     body. Claude then sees plain-text reasoning in its own history, imitates
//     the tag in new replies, and upstream policy may block the request. The
//     Chat path carries no thinking signatures, so this text can never be
//     replayed as real thinking anyway: drop it before conversion.
//  2. Trailing assistant turns: Claude 5.5 rejects assistant prefill with 400.
//     Trim trailing assistant messages after conversion (an empty user message
//     dropped by the converter can also leave one there).
//  3. Upstream refusals: Anthropic reports stop_reason=refusal with no content.
//     The converter mapped it to finish_reason=stop with empty content, which
//     clients render as a silently interrupted reply. Emit a visible notice.

import (
	"encoding/json"
	"errors"
	"regexp"
	"strings"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/tidwall/gjson"
	"go.uber.org/zap"
)

var chatLeakedReasoningPrefix = regexp.MustCompile(`(?s)^\s*<(thinking|think)>.*?</(thinking|think)>\s*`)

const chatRefusalNoticePrefix = "[Sub2API] 本次回复被 Anthropic 上游安全策略拦截"

// stripLeakedReasoningPrefix removes one closed, leading reasoning tag block.
// It never returns an empty string for non-empty input that has other text.
func stripLeakedReasoningPrefix(s string) (string, bool) {
	loc := chatLeakedReasoningPrefix.FindStringIndex(s)
	if loc == nil {
		return s, false
	}
	rest := s[loc[1]:]
	if strings.TrimSpace(rest) == "" {
		return s, false
	}
	return rest, true
}

// sanitizeChatHistoryForAnthropic cleans assistant history in place.
// Returns (reasoning fields dropped, leaked prefixes stripped).
func sanitizeChatHistoryForAnthropic(req *apicompat.ChatCompletionsRequest) (int, int) {
	if req == nil {
		return 0, 0
	}
	dropped, stripped := 0, 0
	for i := range req.Messages {
		m := &req.Messages[i]
		if m.Role != "assistant" {
			continue
		}
		if m.ReasoningContent != "" || m.Reasoning != "" {
			dropped++
			m.ReasoningContent = ""
			m.Reasoning = ""
		}
		if len(m.Content) == 0 {
			continue
		}
		var str string
		if err := json.Unmarshal(m.Content, &str); err == nil {
			if out, ok := stripLeakedReasoningPrefix(str); ok {
				if b, err := json.Marshal(out); err == nil {
					m.Content = b
					stripped++
				}
			}
			continue
		}
		var parts []map[string]any
		if err := json.Unmarshal(m.Content, &parts); err != nil {
			continue
		}
		changed := false
		kept := make([]map[string]any, 0, len(parts))
		for _, p := range parts {
			typ, _ := p["type"].(string)
			if typ == "thinking" || typ == "reasoning" {
				changed = true
				continue
			}
			if text, ok := p["text"].(string); ok {
				if out, ok := stripLeakedReasoningPrefix(text); ok {
					p["text"] = out
					changed = true
					stripped++
				}
			}
			kept = append(kept, p)
		}
		if !changed {
			continue
		}
		if len(kept) == 0 && len(m.ToolCalls) == 0 {
			// Keep the turn alive so user/assistant alternation is preserved.
			kept = append(kept, map[string]any{"type": "text", "text": "(omitted)"})
		}
		if b, err := json.Marshal(kept); err == nil {
			m.Content = b
		}
	}
	return dropped, stripped
}

var errChatEndsWithoutUser = errors.New("conversation must end with a user message")

// trimTrailingAssistantMessages drops trailing assistant turns (no prefill on 5.5).
func trimTrailingAssistantMessages(req *apicompat.AnthropicRequest) (int, error) {
	if req == nil {
		return 0, nil
	}
	n := 0
	for len(req.Messages) > 0 && req.Messages[len(req.Messages)-1].Role == "assistant" {
		req.Messages = req.Messages[:len(req.Messages)-1]
		n++
	}
	if n > 0 && len(req.Messages) == 0 {
		return n, errChatEndsWithoutUser
	}
	return n, nil
}

func chatRefusalNotice(category string) string {
	if category == "" {
		return chatRefusalNoticePrefix + "，未返回内容。请换个说法或开启新对话后重试。"
	}
	return chatRefusalNoticePrefix + "（" + category + "），未返回内容。请换个说法或开启新对话后重试。"
}

// refusalCategoryFromPayload extracts stop_details.category from a raw SSE payload.
func refusalCategoryFromPayload(payload string) string {
	return gjson.Get(payload, "delta.stop_details.category").String()
}

// applyRefusalNoticeToResponse handles the buffered (non-stream) path.
func applyRefusalNoticeToResponse(resp *apicompat.AnthropicResponse, category, requestID string) {
	if resp == nil || resp.StopReason == nil || *resp.StopReason != "refusal" {
		return
	}
	logger.L().Warn("forward_as_cc: upstream refusal",
		zap.String("category", category), zap.String("request_id", requestID), zap.Bool("stream", false))
	resp.Content = append(resp.Content, apicompat.AnthropicContentBlock{Type: "text", Text: chatRefusalNotice(category)})
	resp.StopReason = apicompat.AnthropicStopReasonPtr("end_turn")
}

// chatRefusalTracker rewrites a streamed refusal into a visible text block.
type chatRefusalTracker struct {
	nextIndex int
}

func (t *chatRefusalTracker) observe(event *apicompat.AnthropicStreamEvent) {
	if event != nil && event.Type == "content_block_start" && event.Index != nil && *event.Index >= t.nextIndex {
		t.nextIndex = *event.Index + 1
	}
}

// rewrite returns synthetic events to emit before `event` (and mutates the
// refusal stop_reason to end_turn). nil when nothing to do.
func (t *chatRefusalTracker) rewrite(event *apicompat.AnthropicStreamEvent, payload, requestID string) []apicompat.AnthropicStreamEvent {
	if event == nil || event.Type != "message_delta" || event.Delta == nil || event.Delta.StopReason != "refusal" {
		return nil
	}
	category := refusalCategoryFromPayload(payload)
	logger.L().Warn("forward_as_cc: upstream refusal",
		zap.String("category", category), zap.String("request_id", requestID), zap.Bool("stream", true))
	idx := t.nextIndex
	t.nextIndex++
	event.Delta.StopReason = "end_turn"
	i1, i2, i3 := idx, idx, idx
	return []apicompat.AnthropicStreamEvent{
		{Type: "content_block_start", Index: &i1, ContentBlock: &apicompat.AnthropicContentBlock{Type: "text"}},
		{Type: "content_block_delta", Index: &i2, Delta: &apicompat.AnthropicDelta{Type: "text_delta", Text: chatRefusalNotice(category)}},
		{Type: "content_block_stop", Index: &i3},
	}
}
