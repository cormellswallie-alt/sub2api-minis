package service

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
)

func ccMsg(role string, content any) apicompat.ChatMessage {
	b, _ := json.Marshal(content)
	return apicompat.ChatMessage{Role: role, Content: b}
}

func TestSanitizeDropsReasoningAndLeakedPrefix(t *testing.T) {
	a1 := ccMsg("assistant", "<thinking>draft notes</thinking>\nFinal answer.")
	a1.ReasoningContent = "hidden reasoning"
	a2 := ccMsg("assistant", []map[string]any{
		{"type": "thinking", "thinking": "x"},
		{"type": "text", "text": "<think>y</think> visible"},
	})
	a3 := ccMsg("assistant", "<thinking>only tag</thinking>")
	a4 := ccMsg("assistant", "<thinking>unclosed prefix and text")
	u := ccMsg("user", "<thinking>user text stays</thinking> hi")
	req := &apicompat.ChatCompletionsRequest{Messages: []apicompat.ChatMessage{u, a1, a2, a3, a4}}

	dropped, stripped := sanitizeChatHistoryForAnthropic(req)
	if dropped != 1 || stripped != 2 {
		t.Errorf("dropped=%d stripped=%d", dropped, stripped)
	}
	var s string
	_ = json.Unmarshal(req.Messages[1].Content, &s)
	if s != "Final answer." || req.Messages[1].ReasoningContent != "" {
		t.Errorf("a1=%q rc=%q", s, req.Messages[1].ReasoningContent)
	}
	var parts []map[string]any
	_ = json.Unmarshal(req.Messages[2].Content, &parts)
	if len(parts) != 1 || parts[0]["text"] != "visible" {
		t.Errorf("a2=%v", parts)
	}
	_ = json.Unmarshal(req.Messages[3].Content, &s)
	if s != "<thinking>only tag</thinking>" {
		t.Errorf("tag-only message must stay non-empty, got %q", s)
	}
	_ = json.Unmarshal(req.Messages[4].Content, &s)
	if !strings.HasPrefix(s, "<thinking>unclosed") {
		t.Errorf("unclosed tag must be untouched, got %q", s)
	}
	_ = json.Unmarshal(req.Messages[0].Content, &s)
	if !strings.HasPrefix(s, "<thinking>user") {
		t.Errorf("user message must be untouched, got %q", s)
	}
}

func TestSanitizedHistoryHasNoInlinedReasoning(t *testing.T) {
	a := ccMsg("assistant", "answer")
	a.ReasoningContent = "SECRET-DRAFT"
	req := &apicompat.ChatCompletionsRequest{Model: "claude-opus-5-5",
		Messages: []apicompat.ChatMessage{ccMsg("user", "q1"), a, ccMsg("user", "q2")}}
	sanitizeChatHistoryForAnthropic(req)
	rr, err := apicompat.ChatCompletionsToResponses(req)
	if err != nil {
		t.Fatal(err)
	}
	ar, err := apicompat.ResponsesToAnthropicRequest(rr)
	if err != nil {
		t.Fatal(err)
	}
	b, _ := json.Marshal(ar.Messages)
	if strings.Contains(string(b), "SECRET-DRAFT") || strings.Contains(string(b), "<thinking>") {
		t.Errorf("reasoning leaked into upstream history: %s", b)
	}
	if !strings.Contains(string(b), "answer") {
		t.Errorf("assistant text lost: %s", b)
	}
}

func TestTrimTrailingAssistantMessages(t *testing.T) {
	req := &apicompat.AnthropicRequest{Messages: []apicompat.AnthropicMessage{
		{Role: "user", Content: json.RawMessage(`"q"`)},
		{Role: "assistant", Content: json.RawMessage(`"a"`)},
		{Role: "assistant", Content: json.RawMessage(`"b"`)},
	}}
	n, err := trimTrailingAssistantMessages(req)
	if err != nil || n != 2 || len(req.Messages) != 1 || req.Messages[0].Role != "user" {
		t.Errorf("n=%d err=%v msgs=%v", n, err, req.Messages)
	}
	only := &apicompat.AnthropicRequest{Messages: []apicompat.AnthropicMessage{{Role: "assistant", Content: json.RawMessage(`"a"`)}}}
	if _, err := trimTrailingAssistantMessages(only); err == nil {
		t.Error("assistant-only conversation must error")
	}
	ok := &apicompat.AnthropicRequest{Messages: []apicompat.AnthropicMessage{{Role: "user", Content: json.RawMessage(`"q"`)}}}
	if n, err := trimTrailingAssistantMessages(ok); n != 0 || err != nil {
		t.Errorf("user-tail changed: n=%d err=%v", n, err)
	}
}

func TestBufferedRefusalBecomesVisibleNotice(t *testing.T) {
	resp := &apicompat.AnthropicResponse{ID: "msg_1", Role: "assistant", Model: "claude-opus-5-5",
		StopReason: apicompat.AnthropicStopReasonPtr("refusal")}
	applyRefusalNoticeToResponse(resp, "reasoning_extraction", "r1")
	cc := apicompat.ResponsesToChatCompletions(apicompat.AnthropicToResponsesResponse(resp), "claude-opus-5-5")
	b, _ := json.Marshal(cc)
	if !strings.Contains(string(b), chatRefusalNoticePrefix) || !strings.Contains(string(b), "reasoning_extraction") {
		t.Errorf("notice missing: %s", b)
	}
	normal := &apicompat.AnthropicResponse{StopReason: apicompat.AnthropicStopReasonPtr("end_turn")}
	applyRefusalNoticeToResponse(normal, "", "r2")
	if len(normal.Content) != 0 {
		t.Error("non-refusal response modified")
	}
}

func streamToChatText(t *testing.T, payloads []string) (string, string) {
	t.Helper()
	anth := apicompat.NewAnthropicEventToResponsesState()
	anth.Model = "claude-opus-5-5"
	cc := apicompat.NewResponsesEventToChatState()
	cc.Model = "claude-opus-5-5"
	var tr chatRefusalTracker
	var text strings.Builder
	finish := ""
	emit := func(ev *apicompat.AnthropicStreamEvent) {
		for _, re := range apicompat.AnthropicEventToResponsesEvents(ev, anth) {
			re := re
			for _, ch := range apicompat.ResponsesEventToChatChunks(&re, cc) {
				for _, c := range ch.Choices {
					if c.Delta.Content != nil {
						text.WriteString(*c.Delta.Content)
					}
					if c.FinishReason != nil {
						finish = *c.FinishReason
					}
				}
			}
		}
	}
	for _, p := range payloads {
		var ev apicompat.AnthropicStreamEvent
		if err := json.Unmarshal([]byte(p), &ev); err != nil {
			t.Fatal(err)
		}
		tr.observe(&ev)
		for _, extra := range tr.rewrite(&ev, p, "r") {
			extra := extra
			emit(&extra)
		}
		emit(&ev)
	}
	for _, re := range apicompat.FinalizeAnthropicResponsesStream(anth) {
		re := re
		for _, ch := range apicompat.ResponsesEventToChatChunks(&re, cc) {
			for _, c := range ch.Choices {
				if c.Delta.Content != nil {
					text.WriteString(*c.Delta.Content)
				}
				if c.FinishReason != nil {
					finish = *c.FinishReason
				}
			}
		}
	}
	for _, ch := range apicompat.FinalizeResponsesChatStream(cc) {
		for _, c := range ch.Choices {
			if c.FinishReason != nil {
				finish = *c.FinishReason
			}
		}
	}
	return text.String(), finish
}

func TestStreamRefusalBecomesVisibleNotice(t *testing.T) {
	start := `{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-opus-5-5","content":[],"usage":{"input_tokens":4,"output_tokens":0}}}`
	refusal := `{"type":"message_delta","delta":{"stop_reason":"refusal","stop_details":{"type":"refusal","category":"reasoning_extraction"}},"usage":{"output_tokens":0}}`
	stop := `{"type":"message_stop"}`
	text, finish := streamToChatText(t, []string{start, refusal, stop})
	if !strings.Contains(text, chatRefusalNoticePrefix) || !strings.Contains(text, "reasoning_extraction") {
		t.Errorf("stream notice missing: %q", text)
	}
	if finish != "stop" {
		t.Errorf("finish=%q", finish)
	}

	// After a thinking block at index 0 the notice must use index 1.
	th := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`
	thd := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}`
	ths := `{"type":"content_block_stop","index":0}`
	text, _ = streamToChatText(t, []string{start, th, thd, ths, refusal, stop})
	if !strings.Contains(text, chatRefusalNoticePrefix) {
		t.Errorf("notice after thinking missing: %q", text)
	}

	// Normal streams are untouched.
	tb := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	td := `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`
	ts := `{"type":"content_block_stop","index":0}`
	end := `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}`
	text, finish = streamToChatText(t, []string{start, tb, td, ts, end, stop})
	if text != "hello" || finish != "stop" {
		t.Errorf("normal stream changed: %q %q", text, finish)
	}
}
