package service

// Real-time tool argument streaming on the Chat Completions bridge.
//
// By default Anthropic buffers a tool's input upstream and releases it only
// after the whole JSON is validated, so clients see a long file_write land in
// one burst at the very end. Setting `eager_input_streaming: true` per tool
// makes the upstream stream input_json_delta as it is generated. Only custom
// tools (no type, or type "custom") get the flag; server tools are untouched.
// Disable with SUB2API_CHAT_EAGER_TOOL_STREAMING=false.

import (
	"os"
	"strconv"
	"strings"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

func chatEagerToolStreamingEnabled() bool {
	v := strings.TrimSpace(os.Getenv("SUB2API_CHAT_EAGER_TOOL_STREAMING"))
	if v == "" {
		return true
	}
	b, err := strconv.ParseBool(v)
	return err != nil || b
}

// injectEagerToolInputStreaming returns the body with eager_input_streaming set
// on every custom tool that does not already specify it, plus the count.
func injectEagerToolInputStreaming(body []byte) ([]byte, int) {
	tools := gjson.GetBytes(body, "tools")
	if !tools.IsArray() {
		return body, 0
	}
	n := 0
	for i, tool := range tools.Array() {
		typ := tool.Get("type").String()
		if typ != "" && typ != "custom" {
			continue
		}
		if tool.Get("eager_input_streaming").Exists() {
			continue
		}
		out, err := sjson.SetBytes(body, "tools."+strconv.Itoa(i)+".eager_input_streaming", true)
		if err != nil {
			continue
		}
		body = out
		n++
	}
	return body, n
}
