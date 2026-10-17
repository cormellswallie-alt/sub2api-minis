package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestEagerToolStreamingInjection(t *testing.T) {
	body := []byte(`{"model":"claude-opus-5-5","tools":[` +
		`{"name":"file_write","input_schema":{"type":"object"}},` +
		`{"type":"custom","name":"shell","input_schema":{"type":"object"}},` +
		`{"type":"web_search_20250305","name":"web_search"},` +
		`{"name":"keep","input_schema":{"type":"object"},"eager_input_streaming":false}]}`)
	out, n := injectEagerToolInputStreaming(body)
	if n != 2 {
		t.Errorf("n=%d want 2", n)
	}
	if !gjson.GetBytes(out, "tools.0.eager_input_streaming").Bool() || !gjson.GetBytes(out, "tools.1.eager_input_streaming").Bool() {
		t.Errorf("custom tools not flagged: %s", out)
	}
	if gjson.GetBytes(out, "tools.2.eager_input_streaming").Exists() {
		t.Errorf("server tool must be untouched: %s", out)
	}
	if v := gjson.GetBytes(out, "tools.3.eager_input_streaming"); !v.Exists() || v.Bool() {
		t.Errorf("explicit client value must be kept: %s", out)
	}
	if gjson.GetBytes(out, "tools.0.input_schema.type").String() != "object" {
		t.Errorf("schema damaged: %s", out)
	}
}

func TestEagerToolStreamingNoTools(t *testing.T) {
	body := []byte(`{"model":"claude-opus-5-5","messages":[]}`)
	out, n := injectEagerToolInputStreaming(body)
	if n != 0 || string(out) != string(body) {
		t.Errorf("body without tools changed: %s", out)
	}
}

func TestEagerToolStreamingEnvToggle(t *testing.T) {
	t.Setenv("SUB2API_CHAT_EAGER_TOOL_STREAMING", "")
	if !chatEagerToolStreamingEnabled() {
		t.Error("default must be enabled")
	}
	t.Setenv("SUB2API_CHAT_EAGER_TOOL_STREAMING", "false")
	if chatEagerToolStreamingEnabled() {
		t.Error("false must disable")
	}
}
