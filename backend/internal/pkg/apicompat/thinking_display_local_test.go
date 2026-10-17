package apicompat

import (
	"encoding/json"
	"testing"
)

func TestLocalOpus55ThinkingDisplay(t *testing.T) {
	n := 128000
	req := &ResponsesRequest{Model: "claude-opus-5-5", Input: json.RawMessage(`"hello"`), Stream: true, MaxOutputTokens: &n, Reasoning: &ResponsesReasoning{Effort: "high", Summary: "auto"}}
	out, err := ResponsesToAnthropicRequest(req)
	if err != nil {
		t.Fatal(err)
	}
	if out.Thinking == nil || out.Thinking.Type != "adaptive" || out.Thinking.Display != "summarized" {
		t.Fatalf("wrong thinking: %+v", out.Thinking)
	}
	if out.OutputConfig == nil || out.OutputConfig.Effort != "high" {
		t.Fatal("high effort not preserved")
	}
	if out.MaxTokens != n || !out.Stream {
		t.Fatal("budget or streaming changed")
	}
	b, err := json.Marshal(out)
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if err = json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	if v["thinking"].(map[string]any)["display"] != "summarized" {
		t.Fatal("wire field missing")
	}
}

func TestLocalThinkingDisplayOmitEmpty(t *testing.T) {
	b, err := json.Marshal(AnthropicThinking{Type: "disabled"})
	if err != nil {
		t.Fatal(err)
	}
	var v map[string]any
	if err = json.Unmarshal(b, &v); err != nil {
		t.Fatal(err)
	}
	if _, ok := v["display"]; ok {
		t.Fatal("empty display leaked")
	}
}
