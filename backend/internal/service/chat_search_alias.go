package service

// Web-search model alias for the Chat Completions endpoint.
//
// Clients that cannot add request fields (e.g. Minis model entries) select a
// model id ending in "-search", such as "claude-opus-5-5-search". The request
// body is rewritten at the very start of the handler: the suffix is stripped
// and `{"type":"web_search"}` is appended to tools. Everything downstream
// (scheduling, rate limits, billing, conversion) only ever sees the real model.
// Disable with SUB2API_CHAT_SEARCH_ALIAS=false.

import (
	"os"
	"strconv"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const chatSearchAliasSuffix = "-search"

func chatSearchAliasEnabled() bool {
	v := strings.TrimSpace(os.Getenv("SUB2API_CHAT_SEARCH_ALIAS"))
	if v == "" {
		return true
	}
	b, err := strconv.ParseBool(v)
	return err != nil || b
}

// splitChatSearchAlias returns the base model and whether the alias was used.
func splitChatSearchAlias(model string) (string, bool) {
	m := strings.TrimSpace(model)
	if !strings.HasSuffix(strings.ToLower(m), chatSearchAliasSuffix) {
		return model, false
	}
	base := m[:len(m)-len(chatSearchAliasSuffix)]
	if !strings.HasPrefix(strings.ToLower(base), "claude-") || len(base) <= len("claude-") {
		return model, false
	}
	return base, true
}

// RewriteChatSearchAliasBody rewrites a Chat Completions body that uses a
// "-search" model alias. Returns the (possibly new) body and whether it changed.
func RewriteChatSearchAliasBody(body []byte) ([]byte, bool) {
	if !chatSearchAliasEnabled() {
		return body, false
	}
	base, ok := splitChatSearchAlias(gjson.GetBytes(body, "model").String())
	if !ok {
		return body, false
	}
	out, err := sjson.SetBytes(body, "model", base)
	if err != nil {
		return body, false
	}
	tools := gjson.GetBytes(out, "tools")
	if tools.Exists() && !tools.IsArray() {
		return out, true
	}
	for _, t := range tools.Array() {
		if strings.EqualFold(strings.TrimSpace(t.Get("type").String()), "web_search") {
			return out, true
		}
	}
	if next, err := sjson.SetRawBytes(out, "tools.-1", []byte(`{"type":"web_search"}`)); err == nil {
		out = next
	}
	return out, true
}
