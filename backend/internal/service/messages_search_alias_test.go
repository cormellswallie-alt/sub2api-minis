package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestMessagesSearchAliasRewrite(t *testing.T) {
	body := []byte(`{"model":"claude-opus-5-5-search","max_tokens":64,"messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"name":"file_write","input_schema":{"type":"object"}}]}`)
	out, ok := RewriteMessagesSearchAliasBody(body)
	if !ok {
		t.Fatal("alias not detected")
	}
	if m := gjson.GetBytes(out, "model").String(); m != "claude-opus-5-5" {
		t.Errorf("model=%q", m)
	}
	tools := gjson.GetBytes(out, "tools").Array()
	if len(tools) != 2 || tools[0].Get("name").String() != "file_write" ||
		tools[1].Get("type").String() != "web_search_20250305" || tools[1].Get("name").String() != "web_search" {
		t.Errorf("tools=%s", gjson.GetBytes(out, "tools").Raw)
	}
	if gjson.GetBytes(out, "messages.0.content").String() != "hi" || gjson.GetBytes(out, "max_tokens").Int() != 64 {
		t.Errorf("body damaged: %s", out)
	}
}

func TestMessagesSearchAliasNoToolsAndNoDuplicate(t *testing.T) {
	out, ok := RewriteMessagesSearchAliasBody([]byte(`{"model":"claude-sonnet-5-5-search","messages":[]}`))
	if !ok || gjson.GetBytes(out, "model").String() != "claude-sonnet-5-5" ||
		gjson.GetBytes(out, "tools.#").Int() != 1 || gjson.GetBytes(out, "tools.0.type").String() != "web_search_20250305" {
		t.Errorf("no-tools case: %s", out)
	}
	for _, existing := range []string{
		`{"type":"web_search_20250305","name":"web_search","max_uses":3}`,
		`{"type":"web_search_20260209","name":"web_search"}`,
		`{"name":"web_search","input_schema":{"type":"object"}}`,
	} {
		out, _ = RewriteMessagesSearchAliasBody([]byte(`{"model":"claude-opus-5-5-search","tools":[` + existing + `]}`))
		if gjson.GetBytes(out, "tools.#").Int() != 1 || gjson.GetBytes(out, "model").String() != "claude-opus-5-5" {
			t.Errorf("existing %s duplicated or model kept: %s", existing, out)
		}
	}
}

func TestMessagesSearchAliasIgnoresOtherModels(t *testing.T) {
	for _, m := range []string{"claude-opus-5-5", "gpt-6.1-search", "claude--search", "claude-search", ""} {
		body := []byte(`{"model":"` + m + `"}`)
		out, ok := RewriteMessagesSearchAliasBody(body)
		if ok || string(out) != string(body) {
			t.Errorf("model %q must be untouched, got %s", m, out)
		}
	}
	t.Setenv("SUB2API_CHAT_SEARCH_ALIAS", "false")
	if _, ok := RewriteMessagesSearchAliasBody([]byte(`{"model":"claude-opus-5-5-search"}`)); ok {
		t.Error("env toggle ignored")
	}
}
