package service

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
)

func idleMsg(role string, blocks ...map[string]any) apicompat.AnthropicMessage {
	b, _ := json.Marshal(blocks)
	return apicompat.AnthropicMessage{Role: role, Content: b}
}

func idleText(t string) map[string]any { return map[string]any{"type": "text", "text": t} }

func idleToolUse(id string) map[string]any {
	return map[string]any{"type": "tool_use", "id": id, "name": "shell", "input": map[string]any{"cmd": "ls"}}
}

func idleToolResult(id, out string) map[string]any {
	return map[string]any{"type": "tool_result", "tool_use_id": id, "content": out}
}

// idleConversation: 9 messages; clean user turns at 0, 2, 6, 8; tool_result at 4.
func idleConversation() []apicompat.AnthropicMessage {
	pad := strings.Repeat("x", 1000)
	return []apicompat.AnthropicMessage{
		idleMsg("user", idleText("first question "+pad)),
		idleMsg("assistant", idleText("first answer "+pad)),
		idleMsg("user", idleText("second question about the config "+pad)),
		idleMsg("assistant", idleText("let me look"), idleToolUse("t1")),
		idleMsg("user", idleToolResult("t1", "output "+pad)),
		idleMsg("assistant", idleText("found it "+pad)),
		idleMsg("user", idleText("third question "+pad)),
		idleMsg("assistant", idleText("third answer "+pad)),
		idleMsg("user", idleText("latest question")),
	}
}

func idleTestContext(auth string) *gin.Context {
	gin.SetMode(gin.TestMode)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	if auth != "" {
		c.Request.Header.Set("Authorization", auth)
	}
	return c
}

func idleReq(msgs []apicompat.AnthropicMessage) *apicompat.AnthropicRequest {
	cp := append([]apicompat.AnthropicMessage(nil), msgs...)
	return &apicompat.AnthropicRequest{Model: "claude-opus-5-5", Messages: cp, Stream: true}
}

func idleTestCompactor(t *testing.T) *idleCompactor {
	t.Helper()
	cfg := defaultIdleCompactionConfig()
	cfg.Enabled = true
	return newIdleCompactor(cfg, func(context.Context, *idleCompactionSnapshot, []byte) (*http.Response, error) {
		t.Error("unexpected send")
		return nil, nil
	})
}

func TestIdleCompactionAppliesSummaryWhenPrefixMatches(t *testing.T) {
	ic := idleTestCompactor(t)
	base := idleConversation()
	first := ic.prepare(idleTestContext("Bearer k1"), idleReq(base))
	if first == nil || first.applied != nil {
		t.Fatalf("unexpected first ticket: %+v", first)
	}
	ic.records[first.key] = &idleCompactionRecord{
		Key: first.key, Boundary: 6, PrefixHash: first.origHashes[5], Summary: "SUMMARY-TEXT", CreatedAt: time.Now(),
	}

	next := append(append([]apicompat.AnthropicMessage(nil), base...),
		idleMsg("assistant", idleText("latest answer")),
		idleMsg("user", idleText("new question")),
	)
	c := idleTestContext("Bearer k1")
	req := idleReq(next)
	ticket := ic.prepare(c, req)
	if ticket == nil || ticket.applied == nil || ticket.applied.Boundary != 6 {
		t.Fatalf("summary not applied: %+v", ticket)
	}
	if got, want := len(req.Messages), 2+len(next)-6; got != want {
		t.Fatalf("messages = %d, want %d", got, want)
	}
	if req.Messages[0].Role != "user" || !strings.Contains(string(req.Messages[0].Content), "SUMMARY-TEXT") {
		t.Fatalf("first message is not the summary: %s", req.Messages[0].Content)
	}
	if req.Messages[1].Role != "assistant" {
		t.Fatalf("second message must be the assistant ack")
	}
	for i, m := range req.Messages[2:] {
		if string(m.Content) != string(next[6+i].Content) || m.Role != next[6+i].Role {
			t.Fatalf("tail message %d changed", i)
		}
	}
	if c.Writer.Header().Get(idleCompactionHeader) != "6" {
		t.Fatalf("missing %s header", idleCompactionHeader)
	}
	if len(ticket.origHashes) != len(next) || ticket.origHashes[5] != first.origHashes[5] {
		t.Fatalf("ticket must keep original-space hashes")
	}
}

func TestIdleCompactionSkipsWhenPrefixChangedOrBoundaryNotClean(t *testing.T) {
	ic := idleTestCompactor(t)
	base := idleConversation()
	first := ic.prepare(idleTestContext("Bearer k1"), idleReq(base))
	ic.records[first.key] = &idleCompactionRecord{
		Key: first.key, Boundary: 6, PrefixHash: first.origHashes[5], Summary: "S", CreatedAt: time.Now(),
	}

	edited := append([]apicompat.AnthropicMessage(nil), base...)
	edited[3] = idleMsg("assistant", idleText("edited history"))
	req := idleReq(edited)
	if tk := ic.prepare(idleTestContext("Bearer k1"), req); tk.applied != nil {
		t.Fatalf("summary applied although the prefix changed")
	}
	if len(req.Messages) != len(edited) {
		t.Fatalf("messages were modified")
	}

	ic.records[first.key] = &idleCompactionRecord{
		Key: first.key, Boundary: 4, PrefixHash: first.origHashes[3], Summary: "S", CreatedAt: time.Now(),
	}
	if tk := ic.prepare(idleTestContext("Bearer k1"), idleReq(base)); tk.applied != nil {
		t.Fatalf("summary applied at a tool_result boundary")
	}
}

func TestIdleCompactionPlanPicksCleanUserBoundary(t *testing.T) {
	ic := idleTestCompactor(t)
	tk := ic.prepare(idleTestContext("Bearer k1"), idleReq(idleConversation()))
	snap := &idleCompactionSnapshot{ticket: tk, body: make([]byte, 10000), totalTokens: 10000}

	ic.cfg.KeepTokens = 1
	ic.cfg.MinSummarizedPct = 10
	p, ok := ic.plan(snap)
	if !ok || p.boundary != 8 || p.fromEnd != 1 || p.excerpt != "latest question" {
		t.Fatalf("plan = %+v ok=%v, want boundary 8", p, ok)
	}

	// Enough tail to pass 6 and the tool_result at 4: must land on 2.
	ic.cfg.KeepTokens = 4500
	p, ok = ic.plan(snap)
	if !ok || p.boundary != 2 || !strings.HasPrefix(p.excerpt, "second question") {
		t.Fatalf("plan = %+v ok=%v, want boundary 2", p, ok)
	}

	// An earlier summary at 6 is a floor.
	tk.applied = &idleCompactionRecord{Boundary: 6}
	if p, ok = ic.plan(snap); ok {
		t.Fatalf("plan must not move behind the applied boundary: %+v", p)
	}
	tk.applied = nil

	ic.cfg.KeepTokens = 1
	ic.cfg.MinSummarizedPct = 100
	if p, ok = ic.plan(snap); ok {
		t.Fatalf("plan must respect MinSummarizedPct: %+v", p)
	}
}

const idleTestSSE = "event: message_start\n" +
	`data: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5","usage":{"input_tokens":12,"cache_read_input_tokens":9000,"cache_creation_input_tokens":40,"output_tokens":1}}}` + "\n\n" +
	`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hidden"}}` + "\n\n" +
	`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"SUMMARY: the user asked three questions; config path is /etc/x. "}}` + "\n\n" +
	`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Pending: answer the latest question. Padding so the summary passes the minimum length check of two hundred characters easily, with some extra words added to stay clearly above it."}}` + "\n\n" +
	`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":321}}` + "\n\n" +
	`data: {"type":"message_stop"}` + "\n\n"

func idleUpstreamBody(t *testing.T, msgs []apicompat.AnthropicMessage) []byte {
	t.Helper()
	cc := map[string]any{"type": "ephemeral", "ttl": "1h"}
	out := make([]map[string]any, 0, len(msgs))
	for i, m := range msgs {
		var content any
		_ = json.Unmarshal(m.Content, &content)
		if i == len(msgs)-1 {
			if arr, ok := content.([]any); ok && len(arr) > 0 {
				arr[len(arr)-1].(map[string]any)["cache_control"] = cc
			}
		}
		out = append(out, map[string]any{"role": m.Role, "content": content})
	}
	b, err := json.Marshal(map[string]any{
		"model":    "claude-opus-5-5",
		"stream":   true,
		"system":   []any{map[string]any{"type": "text", "text": "sys", "cache_control": cc}},
		"messages": out,
	})
	if err != nil {
		t.Fatal(err)
	}
	return b
}

func idleUpstreamReq(t *testing.T) *http.Request {
	t.Helper()
	r, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages?beta=true", nil)
	if err != nil {
		t.Fatal(err)
	}
	r.Header.Set("authorization", "Bearer upstream-old")
	r.Header.Set("anthropic-beta", "x")
	return r
}

func idleBigResult() *ForwardResult {
	return &ForwardResult{Usage: ClaudeUsage{InputTokens: 10, CacheReadInputTokens: 9000, CacheCreationInputTokens: 990}}
}

func TestIdleCompactionEndToEndSummarizesAndPersists(t *testing.T) {
	statePath := filepath.Join(t.TempDir(), "idle.json")
	cfg := defaultIdleCompactionConfig()
	cfg.Enabled = true
	cfg.MinTokens = 1000
	cfg.KeepTokens = 4500
	cfg.MinSummarizedPct = 10
	cfg.IdleAfter = 20 * time.Millisecond
	cfg.StatePath = statePath

	sent := make(chan []byte, 1)
	ic := newIdleCompactor(cfg, func(_ context.Context, snap *idleCompactionSnapshot, body []byte) (*http.Response, error) {
		if snap.header.Get("anthropic-beta") != "x" {
			t.Errorf("snapshot headers were not kept")
		}
		sent <- body
		return &http.Response{StatusCode: http.StatusOK, Header: http.Header{}, Body: io.NopCloser(strings.NewReader(idleTestSSE))}, nil
	})

	base := idleConversation()
	c := idleTestContext("Bearer k1")
	tk := ic.prepare(c, idleReq(base))
	upstream := idleUpstreamBody(t, base)
	ic.record(tk, c, idleUpstreamReq(t), upstream, &Account{ID: 7, Concurrency: 1}, "", "oauth", time.Now(), idleBigResult())

	var body []byte
	select {
	case body = <-sent:
	case <-time.After(3 * time.Second):
		t.Fatal("idle job did not run")
	}

	// The upstream request is byte-identical up to the appended messages.
	msgs := gjson.GetBytes(body, "messages").Array()
	if len(msgs) != len(base)+2 {
		t.Fatalf("summary request has %d messages, want %d", len(msgs), len(base)+2)
	}
	origMsgs := gjson.GetBytes(upstream, "messages").Array()
	for i := range origMsgs {
		if msgs[i].Raw != origMsgs[i].Raw {
			t.Fatalf("message %d changed in the summary request", i)
		}
	}
	if gjson.GetBytes(body, "max_tokens").Int() != idleCompactionMinMaxTokens {
		t.Fatalf("summary request max_tokens floor not applied")
	}
	if gjson.GetBytes(body, "system").Raw != gjson.GetBytes(upstream, "system").Raw {
		t.Fatalf("system changed in the summary request")
	}
	if msgs[len(base)].Get("role").String() != "assistant" || msgs[len(base)+1].Get("role").String() != "user" {
		t.Fatalf("appended roles are wrong")
	}
	instr := msgs[len(base)+1].Get("content.0.text").String()
	if !strings.Contains(instr, "second question about the config") || !strings.Contains(instr, "#7") {
		t.Fatalf("instruction does not identify the boundary: %s", instr)
	}

	var rec *idleCompactionRecord
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		ic.mu.Lock()
		rec = ic.records[tk.key]
		_, pending := ic.snaps[tk.key]
		ic.mu.Unlock()
		if rec != nil && !pending {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	if rec == nil || rec.Boundary != 2 || !strings.HasPrefix(rec.Summary, "SUMMARY:") || strings.Contains(rec.Summary, "hidden") {
		t.Fatalf("bad record: %+v", rec)
	}

	// A restarted gateway loads the record and substitutes the next request.
	ic2 := newIdleCompactor(cfg, nil)
	ic2.loadState()
	next := append(append([]apicompat.AnthropicMessage(nil), base...),
		idleMsg("assistant", idleText("latest answer")), idleMsg("user", idleText("new question")))
	req := idleReq(next)
	tk2 := ic2.prepare(idleTestContext("Bearer k1"), req)
	if tk2.applied == nil || len(req.Messages) != 2+len(next)-2 {
		t.Fatalf("persisted summary not applied: applied=%v len=%d", tk2.applied != nil, len(req.Messages))
	}
	if !strings.Contains(string(req.Messages[0].Content), "SUMMARY:") {
		t.Fatalf("summary text missing after reload")
	}
	// Another client credential never sees this summary.
	if tk3 := ic2.prepare(idleTestContext("Bearer other"), idleReq(next)); tk3.applied != nil {
		t.Fatalf("summary leaked across client credentials")
	}
}

func TestIdleCompactionNewRequestCancelsPendingJob(t *testing.T) {
	cfg := defaultIdleCompactionConfig()
	cfg.Enabled = true
	cfg.MinTokens = 1000
	cfg.IdleAfter = 150 * time.Millisecond
	called := make(chan struct{}, 1)
	ic := newIdleCompactor(cfg, func(context.Context, *idleCompactionSnapshot, []byte) (*http.Response, error) {
		called <- struct{}{}
		return nil, io.EOF
	})
	base := idleConversation()
	c := idleTestContext("Bearer k1")
	tk := ic.prepare(c, idleReq(base))
	ic.record(tk, c, idleUpstreamReq(t), idleUpstreamBody(t, base), &Account{ID: 1}, "", "oauth", time.Now(), idleBigResult())
	ic.mu.Lock()
	_, pending := ic.snaps[tk.key]
	ic.mu.Unlock()
	if !pending {
		t.Fatal("large session was not scheduled")
	}

	// The user comes back before the idle point.
	ic.prepare(idleTestContext("Bearer k1"), idleReq(base))
	select {
	case <-called:
		t.Fatal("idle job ran although the conversation resumed")
	case <-time.After(400 * time.Millisecond):
	}

	// A late record() from the older request must not reschedule.
	ic.record(tk, c, idleUpstreamReq(t), idleUpstreamBody(t, base), &Account{ID: 1}, "", "oauth", time.Now(), idleBigResult())
	ic.mu.Lock()
	_, pending = ic.snaps[tk.key]
	ic.mu.Unlock()
	if pending {
		t.Fatal("stale request rescheduled an idle job")
	}
}

func TestIdleCompactionRecordGating(t *testing.T) {
	cfg := defaultIdleCompactionConfig()
	cfg.Enabled = true
	cfg.MinTokens = 200000
	ic := newIdleCompactor(cfg, nil)
	base := idleConversation()
	c := idleTestContext("Bearer k1")
	pending := func(tk *idleCompactionTicket) bool {
		ic.mu.Lock()
		defer ic.mu.Unlock()
		snap, ok := ic.snaps[tk.key]
		if ok && snap.timer != nil {
			snap.timer.Stop()
		}
		return ok
	}

	tk := ic.prepare(c, idleReq(base))
	ic.record(tk, c, idleUpstreamReq(t), idleUpstreamBody(t, base), &Account{ID: 1}, "", "oauth", time.Now(), idleBigResult())
	if pending(tk) {
		t.Fatal("session below MinTokens was scheduled")
	}

	ic.cfg.MinTokens = 1000
	tk = ic.prepare(c, idleReq(base))
	short := []byte(strings.ReplaceAll(string(idleUpstreamBody(t, base)), `"ttl":"1h"`, `"ttl":"5m"`))
	ic.record(tk, c, idleUpstreamReq(t), short, &Account{ID: 1}, "", "oauth", time.Now(), idleBigResult())
	if pending(tk) {
		t.Fatal("5m cache session was scheduled without ALLOW_5M")
	}

	tk = ic.prepare(c, idleReq(base))
	res := idleBigResult()
	res.ClientDisconnect = true
	ic.record(tk, c, idleUpstreamReq(t), idleUpstreamBody(t, base), &Account{ID: 1}, "", "oauth", time.Now(), res)
	if pending(tk) {
		t.Fatal("disconnected request was scheduled")
	}

	tk = ic.prepare(c, idleReq(base))
	ic.record(tk, c, idleUpstreamReq(t), idleUpstreamBody(t, base), &Account{ID: 1}, "", "oauth", time.Now(), idleBigResult())
	if !pending(tk) {
		t.Fatal("eligible session was not scheduled")
	}
}

func TestIdleCompactionHelpers(t *testing.T) {
	if got := idleDetectCacheTTL([]byte(`{"system":[{"type":"text","text":"s","cache_control":{"type":"ephemeral","ttl":"1h"}}],"messages":[]}`)); got != time.Hour {
		t.Fatalf("ttl = %v, want 1h", got)
	}
	if got := idleDetectCacheTTL([]byte(`{"tools":[{"name":"a","cache_control":{"type":"ephemeral"}}],"messages":[]}`)); got != 5*time.Minute {
		t.Fatalf("ttl = %v, want 5m", got)
	}
	if got := idleDetectCacheTTL([]byte(`{"messages":[{"role":"user","content":"hi"}]}`)); got != 0 {
		t.Fatalf("ttl = %v, want 0", got)
	}

	ic := newIdleCompactor(defaultIdleCompactionConfig(), nil)
	if d := ic.idleDelay(time.Hour); d != 54*time.Minute {
		t.Fatalf("1h delay = %v, want 54m", d)
	}
	if d := ic.idleDelay(5 * time.Minute); d != 4*time.Minute+30*time.Second {
		t.Fatalf("5m delay = %v, want 4m30s", d)
	}
	ic.cfg.IdleAfter = time.Minute
	if d := ic.idleDelay(time.Hour); d != time.Minute {
		t.Fatalf("override delay = %v", d)
	}

	errSSE := `data: {"type":"error","error":{"type":"overloaded_error","message":"busy"}}` + "\n\n"
	if _, _, _, err := parseIdleCompactionStream(strings.NewReader(errSSE), 0); err == nil {
		t.Fatal("stream error was not reported")
	}
	text, usage, stop, err := parseIdleCompactionStream(strings.NewReader(idleTestSSE), 0)
	if err != nil || stop != "end_turn" || usage.CacheReadInputTokens != 9000 || usage.OutputTokens != 321 || strings.Contains(text, "hidden") {
		t.Fatalf("parse = %q %+v %q %v", text, usage, stop, err)
	}

	if _, err := buildIdleCompactionBody([]byte(`{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}`), idleCompactionPlan{}); err == nil {
		t.Fatal("body ending with assistant must be rejected")
	}

	first := idleMsg("user", idleText("hello"))
	h1, h2 := http.Header{}, http.Header{}
	h1.Set("Authorization", "Bearer a")
	h2.Set("Authorization", "Bearer b")
	if idleCompactionKey(h1, "m", first) == idleCompactionKey(h2, "m", first) {
		t.Fatal("key must depend on the client credential")
	}
	if idleCompactionKey(h1, "m", first) == idleCompactionKey(h1, "n", first) {
		t.Fatal("key must depend on the model")
	}

	ok, ex := idleCleanUserMessage(apicompat.AnthropicMessage{Role: "user", Content: json.RawMessage(`"  plain   string  "`)})
	if !ok || ex != "plain string" {
		t.Fatalf("string content: %v %q", ok, ex)
	}
	if ok, _ := idleCleanUserMessage(idleMsg("user", idleToolResult("t", "x"))); ok {
		t.Fatal("tool_result turn must not be clean")
	}
	if ok, ex := idleCleanUserMessage(idleMsg("user", idleText(strings.Repeat("长", 300)))); !ok || !strings.HasSuffix(ex, "…") || len([]rune(ex)) != 121 {
		t.Fatalf("excerpt not truncated: %d", len([]rune(ex)))
	}
}
