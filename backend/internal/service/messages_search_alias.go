package service

// Web-search model alias for the Anthropic Messages endpoint (/v1/messages and
// /v1/messages/count_tokens).
//
// Same contract as the Chat Completions alias (chat_search_alias.go): a model
// id ending in "-search", e.g. "claude-opus-5-5-search", is rewritten to the
// base model and Anthropic's server-side web search tool is appended to tools.
// Downstream code (scheduling, rate limits, billing) only sees the real model.
// Shares the SUB2API_CHAT_SEARCH_ALIAS=false kill switch.

import (
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const messagesWebSearchTool = `{"type":"web_search_20250305","name":"web_search"}`

// hasAnthropicWebSearchTool reports whether tools already declare a web search
// server tool or any tool named "web_search" (a second one would be a name clash).
func hasAnthropicWebSearchTool(tools gjson.Result) bool {
	for _, t := range tools.Array() {
		typ := strings.ToLower(strings.TrimSpace(t.Get("type").String()))
		if strings.HasPrefix(typ, "web_search") {
			return true
		}
		if strings.EqualFold(strings.TrimSpace(t.Get("name").String()), "web_search") {
			return true
		}
	}
	return false
}

// RewriteMessagesSearchAliasBody rewrites an Anthropic Messages body that uses
// a "-search" model alias. Returns the (possibly new) body and whether it changed.
func RewriteMessagesSearchAliasBody(body []byte) ([]byte, bool) {
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
	if hasAnthropicWebSearchTool(tools) {
		return out, true
	}
	if next, err := sjson.SetRawBytes(out, "tools.-1", []byte(messagesWebSearchTool)); err == nil {
		out = next
	}
	return out, true
}
