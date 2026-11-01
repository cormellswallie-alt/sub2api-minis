package service

import (
	"testing"

	"github.com/tidwall/gjson"
)

func TestSearchAliasRewrite(t *testing.T) {
	body := []byte(`{"model":"claude-opus-5-5-search","messages":[{"role":"user","content":"hi"}],` +
		`"tools":[{"type":"function","function":{"name":"file_write","parameters":{"type":"object"}}}]}`)
	out, ok := RewriteChatSearchAliasBody(body)
	if !ok {
		t.Fatal("alias not detected")
	}
	if m := gjson.GetBytes(out, "model").String(); m != "claude-opus-5-5" {
		t.Errorf("model=%q", m)
	}
	tools := gjson.GetBytes(out, "tools").Array()
	if len(tools) != 2 || tools[0].Get("function.name").String() != "file_write" || tools[1].Get("type").String() != "web_search" {
		t.Errorf("tools=%s", gjson.GetBytes(out, "tools").Raw)
	}
	if gjson.GetBytes(out, "messages.0.content").String() != "hi" {
		t.Errorf("messages damaged: %s", out)
	}
}

func TestSearchAliasNoToolsAndNoDuplicate(t *testing.T) {
	out, ok := RewriteChatSearchAliasBody([]byte(`{"model":"claude-sonnet-5-5-search","messages":[]}`))
	if !ok || gjson.GetBytes(out, "model").String() != "claude-sonnet-5-5" ||
		gjson.GetBytes(out, "tools.#").Int() != 1 || gjson.GetBytes(out, "tools.0.type").String() != "web_search" {
		t.Errorf("no-tools case: %s", out)
	}
	out, _ = RewriteChatSearchAliasBody([]byte(`{"model":"claude-opus-5-5-search","tools":[{"type":"web_search"}]}`))
	if gjson.GetBytes(out, "tools.#").Int() != 1 {
		t.Errorf("duplicated web_search: %s", out)
	}
}

func TestSearchAliasIgnoresOtherModels(t *testing.T) {
	for _, m := range []string{"claude-opus-5-5", "gpt-6.1-search", "claude--search", "claude-search", ""} {
		body := []byte(`{"model":"` + m + `"}`)
		out, ok := RewriteChatSearchAliasBody(body)
		if ok || string(out) != string(body) {
			t.Errorf("model %q must be untouched, got %s", m, out)
		}
	}
	t.Setenv("SUB2API_CHAT_SEARCH_ALIAS", "false")
	if _, ok := RewriteChatSearchAliasBody([]byte(`{"model":"claude-opus-5-5-search"}`)); ok {
		t.Error("env toggle ignored")
	}
}
