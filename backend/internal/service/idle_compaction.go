package service

// Idle compaction for the Chat Completions -> Anthropic bridge.
//
// Claude Code 2.1.299 added "idle compaction": a large conversation left idle
// shortly before its one-hour prompt cache expires is compacted, so the next
// message does not have to re-create the whole cache. Clients that reach
// Sub2API through /v1/chat/completions (e.g. Minis) always resend the full
// history and cannot be told to compact, so this file does it at the gateway:
//
//  1. Every successful request of a large session is snapshotted (the exact
//     upstream body + headers). A timer fires at cacheTTL - margin.
//  2. When the session is still idle at that point, the gateway replays the
//     snapshot while the cache is still warm, with one extra instruction asking
//     the model to summarise everything before a clean user-turn boundary.
//     The prefix is byte-identical, so the summary request is a cache read.
//  3. The summary is stored with a chained hash of the original messages
//     before the boundary. When the client later resends that same prefix,
//     the gateway replaces it with [summary, ack] and forwards the rest
//     verbatim. If the prefix no longer matches, nothing is substituted.
//
// Everything is off unless SUB2API_IDLE_COMPACTION_ENABLED=true.

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/Wei-Shaw/sub2api/internal/pkg/apicompat"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/Wei-Shaw/sub2api/internal/pkg/tlsfingerprint"
	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"go.uber.org/zap"
)

const (
	idleCompactionStateVersion = 1
	idleCompactionMinMaxTokens = 16384
	idleCompactionHeader       = "X-Sub2API-Idle-Compacted"
	idleCompactionSummaryOpen  = "<idle-compaction-summary>"
	idleCompactionSummaryClose = "</idle-compaction-summary>"
	idleCompactionPlaceholder  = "(Reply omitted.)"
	idleCompactionAck          = "Understood. I have the summary of the earlier conversation and will continue from it together with the messages that follow."
)

// IdleCompactionConfig is read once from the environment.
type IdleCompactionConfig struct {
	Enabled          bool
	MinTokens        int           // only sessions whose context reached this size are compacted
	IdleAfter        time.Duration // 0 = derive from cache TTL (ttl - margin)
	AllowShortTTL    bool          // also compact sessions whose cache TTL is only 5m
	KeepTokens       int           // newest context kept verbatim (approximate)
	SummaryMaxChars  int           // reject summaries longer than this
	SummaryMinChars  int           // reject summaries shorter than this
	StatePath        string        // "" = memory only
	MaxSessions      int           // concurrently tracked idle snapshots
	MaxRecords       int           // stored summaries
	RecordMaxAge     time.Duration // stored summaries expire after this
	JobTimeout       time.Duration
	MinSummarizedPct int // the summarised part must be at least this % of the context
}

func defaultIdleCompactionConfig() IdleCompactionConfig {
	return IdleCompactionConfig{
		MinTokens:        200000,
		KeepTokens:       40000,
		SummaryMaxChars:  120000,
		SummaryMinChars:  200,
		MaxSessions:      8,
		MaxRecords:       64,
		RecordMaxAge:     7 * 24 * time.Hour,
		JobTimeout:       5 * time.Minute,
		MinSummarizedPct: 40,
	}
}

func loadIdleCompactionConfigFromEnv() IdleCompactionConfig {
	cfg := defaultIdleCompactionConfig()
	cfg.Enabled = idleEnvBool("SUB2API_IDLE_COMPACTION_ENABLED", false)
	cfg.MinTokens = idleEnvInt("SUB2API_IDLE_COMPACTION_MIN_TOKENS", cfg.MinTokens)
	cfg.KeepTokens = idleEnvInt("SUB2API_IDLE_COMPACTION_KEEP_TOKENS", cfg.KeepTokens)
	cfg.AllowShortTTL = idleEnvBool("SUB2API_IDLE_COMPACTION_ALLOW_5M", false)
	cfg.MaxSessions = idleEnvInt("SUB2API_IDLE_COMPACTION_MAX_SESSIONS", cfg.MaxSessions)
	cfg.MinSummarizedPct = idleEnvInt("SUB2API_IDLE_COMPACTION_MIN_SUMMARIZED_PCT", cfg.MinSummarizedPct)
	if v := strings.TrimSpace(os.Getenv("SUB2API_IDLE_COMPACTION_AFTER")); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			cfg.IdleAfter = d
		}
	}
	if v, ok := os.LookupEnv("SUB2API_IDLE_COMPACTION_STATE_FILE"); ok {
		cfg.StatePath = strings.TrimSpace(v)
	} else if dir := strings.TrimSpace(os.Getenv("DATA_DIR")); dir != "" {
		cfg.StatePath = filepath.Join(dir, "idle_compaction.json")
	}
	return cfg
}

func idleEnvBool(name string, def bool) bool {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	b, err := strconv.ParseBool(v)
	if err != nil {
		return def
	}
	return b
}

func idleEnvInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n < 0 {
		return def
	}
	return n
}

var (
	chatCacheTTL1hOnce    sync.Once
	chatCacheTTL1hAllowed bool
)

// chatCompletionsCacheTTL1hAllowed gates honoring the admin "1h cache TTL"
// setting on the Chat Completions bridge. Default on; set
// SUB2API_CHAT_CACHE_TTL_1H=false to keep the old 5m behaviour.
func chatCompletionsCacheTTL1hAllowed() bool {
	chatCacheTTL1hOnce.Do(func() {
		chatCacheTTL1hAllowed = idleEnvBool("SUB2API_CHAT_CACHE_TTL_1H", true)
	})
	return chatCacheTTL1hAllowed
}

// idleCompactionRecord is a stored summary for one conversation prefix.
type idleCompactionRecord struct {
	Key          string    `json:"key"`
	Boundary     int       `json:"boundary"`    // number of original messages replaced
	PrefixHash   string    `json:"prefix_hash"` // chained hash of original messages [0, Boundary)
	Summary      string    `json:"summary"`
	CreatedAt    time.Time `json:"created_at"`
	SourceTokens int       `json:"source_tokens"`
}

// idleCompactionTicket carries per-request state from prepare to record.
type idleCompactionTicket struct {
	key        string
	gen        uint64
	origHashes []string // origHashes[i] = chained hash of original messages [0, i]
	origSizes  []int
	cleanUser  []bool
	excerpts   []string
	applied    *idleCompactionRecord
}

type idleCompactionSnapshot struct {
	ticket      *idleCompactionTicket
	url         string
	header      http.Header
	body        []byte
	account     *Account
	proxyURL    string
	tokenType   string
	toolRewrite *ToolNameRewrite
	startedAt   time.Time
	cacheTTL    time.Duration
	totalTokens int
	timer       *time.Timer
	cancel      context.CancelFunc
}

type idleCompactionSendFunc func(ctx context.Context, snap *idleCompactionSnapshot, body []byte) (*http.Response, error)

type idleCompactor struct {
	cfg  IdleCompactionConfig
	send idleCompactionSendFunc
	now  func() time.Time

	mu      sync.Mutex
	seq     uint64
	gens    map[string]uint64
	snaps   map[string]*idleCompactionSnapshot
	order   []string
	records map[string]*idleCompactionRecord

	saveMu sync.Mutex
}

func newIdleCompactor(cfg IdleCompactionConfig, send idleCompactionSendFunc) *idleCompactor {
	return &idleCompactor{
		cfg:     cfg,
		send:    send,
		now:     time.Now,
		gens:    make(map[string]uint64),
		snaps:   make(map[string]*idleCompactionSnapshot),
		records: make(map[string]*idleCompactionRecord),
	}
}

var (
	idleCompactorOnce     sync.Once
	idleCompactorInstance *idleCompactor
)

// idleCompactorFor returns the process-wide compactor, or nil when disabled.
func idleCompactorFor(s *GatewayService) *idleCompactor {
	idleCompactorOnce.Do(func() {
		cfg := loadIdleCompactionConfigFromEnv()
		if !cfg.Enabled || s == nil {
			return
		}
		ic := newIdleCompactor(cfg, s.sendIdleCompactionRequest)
		ic.loadState()
		idleCompactorInstance = ic
		logger.L().Info("idle_compaction.enabled",
			zap.Int("min_tokens", cfg.MinTokens),
			zap.Int("keep_tokens", cfg.KeepTokens),
			zap.Duration("idle_after", cfg.IdleAfter),
			zap.Bool("allow_5m", cfg.AllowShortTTL),
			zap.String("state_path", cfg.StatePath),
			zap.Int("records", len(ic.records)),
		)
	})
	return idleCompactorInstance
}

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

// prepare is called for every bridged request. It cancels any pending idle job
// of the same conversation and, when a stored summary matches the request's
// prefix, rewrites req.Messages in place.
func (ic *idleCompactor) prepare(c *gin.Context, req *apicompat.AnthropicRequest) *idleCompactionTicket {
	if ic == nil || req == nil || len(req.Messages) == 0 {
		return nil
	}
	var reqHeader http.Header
	if c != nil && c.Request != nil {
		reqHeader = c.Request.Header
	}
	n := len(req.Messages)
	t := &idleCompactionTicket{
		key:        idleCompactionKey(reqHeader, req.Model, req.Messages[0]),
		origHashes: make([]string, n),
		origSizes:  make([]int, n),
		cleanUser:  make([]bool, n),
		excerpts:   make([]string, n),
	}
	prev := ""
	for i, m := range req.Messages {
		prev = idleChainHash(prev, m)
		t.origHashes[i] = prev
		t.origSizes[i] = len(m.Role) + len(m.Content) + 16
		t.cleanUser[i], t.excerpts[i] = idleCleanUserMessage(m)
	}

	ic.mu.Lock()
	ic.seq++
	t.gen = ic.seq
	ic.gens[t.key] = t.gen
	ic.dropSnapLocked(t.key)
	rec := ic.records[t.key]
	if rec != nil && rec.Boundary > 0 && rec.Boundary < n &&
		t.origHashes[rec.Boundary-1] == rec.PrefixHash && t.cleanUser[rec.Boundary] {
		cp := *rec
		t.applied = &cp
	}
	ic.pruneGensLocked(t.key)
	ic.mu.Unlock()

	if t.applied != nil {
		req.Messages = idleApplySummary(req.Messages, t.applied)
		if c != nil {
			c.Header(idleCompactionHeader, strconv.Itoa(t.applied.Boundary))
		}
	}
	return t
}

func idleApplySummary(msgs []apicompat.AnthropicMessage, rec *idleCompactionRecord) []apicompat.AnthropicMessage {
	userText := "This session is being continued from an earlier conversation that was compacted while idle. " +
		"The summary below covers the earlier part; the messages after it are verbatim.\n\n" +
		idleCompactionSummaryOpen + "\n" + rec.Summary + "\n" + idleCompactionSummaryClose
	out := make([]apicompat.AnthropicMessage, 0, len(msgs)-rec.Boundary+2)
	out = append(out,
		apicompat.AnthropicMessage{Role: "user", Content: idleTextContent(userText)},
		apicompat.AnthropicMessage{Role: "assistant", Content: idleTextContent(idleCompactionAck)},
	)
	return append(out, msgs[rec.Boundary:]...)
}

func idleTextContent(text string) json.RawMessage {
	b, _ := json.Marshal([]map[string]string{{"type": "text", "text": text}})
	return b
}

// record stores a snapshot of a successful request and schedules the idle job.
func (ic *idleCompactor) record(t *idleCompactionTicket, c *gin.Context, upstreamReq *http.Request, body []byte,
	account *Account, proxyURL, tokenType string, startedAt time.Time, result *ForwardResult) {
	if ic == nil || t == nil || upstreamReq == nil || account == nil || result == nil || len(body) == 0 {
		return
	}
	if result.ClientDisconnect {
		return
	}
	u := result.Usage
	total := u.InputTokens + u.CacheReadInputTokens + u.CacheCreationInputTokens
	if total < ic.cfg.MinTokens {
		return
	}
	ttl := idleDetectCacheTTL(body)
	if ttl <= 0 || (ttl < time.Hour && !ic.cfg.AllowShortTTL && ic.cfg.IdleAfter == 0) {
		return
	}
	snap := &idleCompactionSnapshot{
		ticket:      t,
		url:         upstreamReq.URL.String(),
		header:      upstreamReq.Header.Clone(),
		body:        append([]byte(nil), body...),
		account:     account,
		proxyURL:    proxyURL,
		tokenType:   tokenType,
		startedAt:   startedAt,
		cacheTTL:    ttl,
		totalTokens: total,
	}
	if c != nil {
		snap.toolRewrite = toolNameRewriteFromContext(c)
	}
	delay := ic.idleDelay(ttl)
	wait := delay - ic.now().Sub(startedAt)
	if wait < 10*time.Millisecond {
		wait = 10 * time.Millisecond
	}

	ic.mu.Lock()
	defer ic.mu.Unlock()
	if ic.gens[t.key] != t.gen {
		return // a newer request of the same conversation already started
	}
	ic.dropSnapLocked(t.key)
	ic.snaps[t.key] = snap
	ic.order = append(ic.order, t.key)
	for len(ic.snaps) > ic.cfg.MaxSessions && len(ic.order) > 0 {
		oldest := ic.order[0]
		if oldest == t.key && len(ic.snaps) == 1 {
			break
		}
		ic.dropSnapLocked(oldest)
	}
	key, gen := t.key, t.gen
	snap.timer = time.AfterFunc(wait, func() { ic.fire(key, gen) })
}

func (ic *idleCompactor) idleDelay(ttl time.Duration) time.Duration {
	if ic.cfg.IdleAfter > 0 {
		return ic.cfg.IdleAfter
	}
	margin := ttl / 10
	if margin > 6*time.Minute {
		margin = 6 * time.Minute
	}
	if margin < 30*time.Second {
		margin = 30 * time.Second
	}
	return ttl - margin
}

// dropSnapLocked removes a pending snapshot and stops its timer / running job.
func (ic *idleCompactor) dropSnapLocked(key string) {
	if snap, ok := ic.snaps[key]; ok {
		if snap.timer != nil {
			snap.timer.Stop()
		}
		if snap.cancel != nil {
			snap.cancel()
		}
		delete(ic.snaps, key)
	}
	if len(ic.order) == 0 {
		return
	}
	kept := ic.order[:0]
	for _, k := range ic.order {
		if k != key {
			kept = append(kept, k)
		}
	}
	ic.order = kept
}

func (ic *idleCompactor) pruneGensLocked(current string) {
	if len(ic.gens) <= 4096 {
		return
	}
	for k := range ic.gens {
		if k == current {
			continue
		}
		if _, ok := ic.snaps[k]; ok {
			continue
		}
		if _, ok := ic.records[k]; ok {
			continue
		}
		delete(ic.gens, k)
	}
}

// ---------------------------------------------------------------------------
// Idle job
// ---------------------------------------------------------------------------

func (ic *idleCompactor) fire(key string, gen uint64) {
	ic.mu.Lock()
	snap := ic.snaps[key]
	if snap == nil || snap.ticket.gen != gen || ic.gens[key] != gen {
		ic.mu.Unlock()
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), ic.cfg.JobTimeout)
	snap.cancel = cancel
	ic.mu.Unlock()
	defer cancel()
	ic.run(ctx, key, gen, snap)
}

func (ic *idleCompactor) run(ctx context.Context, key string, gen uint64, snap *idleCompactionSnapshot) {
	log := logger.L().With(zap.String("idle_key", key[:12]), zap.Int("context_tokens", snap.totalTokens))
	defer ic.release(key, gen)

	plan, ok := ic.plan(snap)
	if !ok {
		log.Info("idle_compaction.skip", zap.String("reason", "no_clean_boundary_or_too_small"))
		return
	}
	body, err := buildIdleCompactionBody(snap.body, plan)
	if err != nil {
		log.Warn("idle_compaction.build_failed", zap.Error(err))
		return
	}
	started := ic.now()
	resp, err := ic.send(ctx, snap, body)
	if err != nil {
		log.Warn("idle_compaction.send_failed", zap.Error(err))
		return
	}
	defer func() { _ = resp.Body.Close() }()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		log.Warn("idle_compaction.upstream_error", zap.Int("status", resp.StatusCode),
			zap.String("message", sanitizeUpstreamErrorMessage(extractUpstreamErrorMessage(raw))))
		return
	}
	var reader io.Reader = resp.Body
	if strings.EqualFold(resp.Header.Get("Content-Encoding"), "gzip") {
		gz, gzErr := gzip.NewReader(resp.Body)
		if gzErr != nil {
			log.Warn("idle_compaction.gzip_failed", zap.Error(gzErr))
			return
		}
		defer func() { _ = gz.Close() }()
		reader = gz
	}
	summary, usage, stop, err := parseIdleCompactionStream(reader, ic.cfg.SummaryMaxChars*2)
	if err != nil {
		log.Warn("idle_compaction.stream_failed", zap.Error(err))
		return
	}
	if snap.toolRewrite != nil {
		summary = string(restoreToolNamesInBytes([]byte(summary), snap.toolRewrite))
	}
	summary = strings.TrimSpace(summary)
	if stop == "max_tokens" || len(summary) < ic.cfg.SummaryMinChars || len(summary) > ic.cfg.SummaryMaxChars {
		log.Warn("idle_compaction.summary_rejected", zap.String("stop_reason", stop), zap.Int("chars", len(summary)))
		return
	}
	rec := &idleCompactionRecord{
		Key:          key,
		Boundary:     plan.boundary,
		PrefixHash:   snap.ticket.origHashes[plan.boundary-1],
		Summary:      summary,
		CreatedAt:    ic.now().UTC(),
		SourceTokens: snap.totalTokens,
	}
	ic.mu.Lock()
	current := ic.gens[key] == gen
	if current {
		ic.records[key] = rec
		ic.pruneRecordsLocked()
	}
	ic.mu.Unlock()
	if !current {
		log.Info("idle_compaction.discarded", zap.String("reason", "conversation_resumed"))
		return
	}
	ic.saveState()
	log.Info("idle_compaction.done",
		zap.Int("boundary", plan.boundary),
		zap.Int("messages", len(snap.ticket.origHashes)),
		zap.Int("summarized_tokens_est", plan.summarizedTokens),
		zap.Int("summary_chars", len(summary)),
		zap.Int("cache_read", usage.CacheReadInputTokens),
		zap.Int("cache_creation", usage.CacheCreationInputTokens),
		zap.Int("input", usage.InputTokens),
		zap.Int("output", usage.OutputTokens),
		zap.Duration("took", ic.now().Sub(started)),
	)
}

func (ic *idleCompactor) release(key string, gen uint64) {
	ic.mu.Lock()
	defer ic.mu.Unlock()
	if snap := ic.snaps[key]; snap != nil && snap.ticket.gen == gen {
		snap.cancel = nil // the job is finishing; do not cancel ourselves
		ic.dropSnapLocked(key)
	}
}

// ---------------------------------------------------------------------------
// Planning and request body
// ---------------------------------------------------------------------------

type idleCompactionPlan struct {
	boundary         int    // original-space index of the first verbatim message
	fromEnd          int    // position of the boundary message counted from the end (last = 1)
	excerpt          string // leading text of the boundary message
	hasPrevious      bool   // the snapshot already carried an earlier summary
	summarizedTokens int
}

// plan picks the newest clean user turn that still leaves about KeepTokens of
// context verbatim, and rejects plans that would summarise too little.
func (ic *idleCompactor) plan(snap *idleCompactionSnapshot) (idleCompactionPlan, bool) {
	t := snap.ticket
	n := len(t.origHashes)
	if n < 3 || len(snap.body) == 0 || snap.totalTokens <= 0 {
		return idleCompactionPlan{}, false
	}
	perByte := float64(snap.totalTokens) / float64(len(snap.body))
	prev := 0
	if t.applied != nil {
		prev = t.applied.Boundary
	}
	tail := 0.0
	boundary := -1
	for i := n - 1; i > prev; i-- {
		tail += float64(t.origSizes[i]) * perByte
		if t.cleanUser[i] && int(tail) >= ic.cfg.KeepTokens {
			boundary = i
			break
		}
	}
	if boundary <= prev || boundary < 2 {
		return idleCompactionPlan{}, false
	}
	summarized := snap.totalTokens - int(tail)
	if summarized*100 < snap.totalTokens*ic.cfg.MinSummarizedPct {
		return idleCompactionPlan{}, false
	}
	return idleCompactionPlan{
		boundary:         boundary,
		fromEnd:          n - boundary,
		excerpt:          t.excerpts[boundary],
		hasPrevious:      t.applied != nil,
		summarizedTokens: summarized,
	}, true
}

// buildIdleCompactionBody appends [placeholder assistant, instruction user] to
// the exact upstream body of the last request. Everything before the appended
// messages stays byte-identical, so the request reads the warm cache.
func buildIdleCompactionBody(body []byte, plan idleCompactionPlan) ([]byte, error) {
	msgs := gjson.GetBytes(body, "messages")
	if !msgs.IsArray() {
		return nil, fmt.Errorf("upstream body has no messages")
	}
	arr := msgs.Array()
	if len(arr) == 0 || arr[len(arr)-1].Get("role").String() != "user" {
		return nil, fmt.Errorf("last upstream message is not a user turn")
	}
	assistant, err := json.Marshal(map[string]any{
		"role":    "assistant",
		"content": []map[string]string{{"type": "text", "text": idleCompactionPlaceholder}},
	})
	if err != nil {
		return nil, err
	}
	user, err := json.Marshal(map[string]any{
		"role":    "user",
		"content": []map[string]string{{"type": "text", "text": idleCompactionInstruction(plan)}},
	})
	if err != nil {
		return nil, err
	}
	out := append([]byte(nil), body...)
	// max_tokens is not part of the cached prefix. A small client value would
	// leave no room after thinking, so give the summary a sane floor.
	if mt := gjson.GetBytes(out, "max_tokens"); !mt.Exists() || mt.Int() < idleCompactionMinMaxTokens {
		if out, err = sjson.SetBytes(out, "max_tokens", idleCompactionMinMaxTokens); err != nil {
			return nil, err
		}
	}
	if out, err = sjson.SetRawBytes(out, "messages.-1", assistant); err != nil {
		return nil, err
	}
	if out, err = sjson.SetRawBytes(out, "messages.-1", user); err != nil {
		return nil, err
	}
	return out, nil
}

func idleCompactionInstruction(plan idleCompactionPlan) string {
	// Phrased as an ordinary user request (like Claude Code's own compaction
	// prompt). A "maintenance request, not from the user" framing reads like a
	// prompt injection and gets refused.
	var b strings.Builder
	b.WriteString("Please pause the task for a moment: this conversation is getting long, so I need a written summary ")
	b.WriteString("of its earlier part. Reply with text only and do not use any tools.\n\n")
	b.WriteString("Summarise everything that came BEFORE ")
	if plan.excerpt != "" {
		fmt.Fprintf(&b, "my message that starts with %q (message #%d counted from the end, before this request). ", plan.excerpt, plan.fromEnd)
	} else {
		fmt.Fprintf(&b, "message #%d counted from the end of the conversation (before this request). ", plan.fromEnd)
	}
	b.WriteString("That message and everything after it stays available as-is, so leave it out.\n")
	if plan.hasPrevious {
		b.WriteString("The conversation starts with an earlier summary inside " + idleCompactionSummaryOpen +
			" tags; fold all of its content into the new summary so nothing from it is lost.\n")
	}
	b.WriteString("\nThe summary should let you continue the work without the earlier messages. Cover, as applicable:\n")
	b.WriteString("1. My requests, goals, preferences and constraints, quoting key wording.\n")
	b.WriteString("2. Decisions made and why; approaches rejected.\n")
	b.WriteString("3. Files, paths, commands, configuration values, identifiers, URLs and numbers that matter later, copied exactly.\n")
	b.WriteString("4. Errors met and how they were fixed.\n")
	b.WriteString("5. Work completed, work in progress, open questions and pending tasks.\n\n")
	b.WriteString("Write in the main language of the conversation. Be dense and factual, with no preamble or closing remarks. ")
	b.WriteString("Output only the summary.")
	return b.String()
}

// parseIdleCompactionStream collects text deltas and usage from an Anthropic SSE stream.
func parseIdleCompactionStream(r io.Reader, maxChars int) (string, ClaudeUsage, string, error) {
	var (
		text  strings.Builder
		usage ClaudeUsage
		stop  string
	)
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	for sc.Scan() {
		data, ok := parseAnthropicSSEField(sc.Text(), "data")
		if !ok || data == "" || data == "[DONE]" {
			continue
		}
		var ev apicompat.AnthropicStreamEvent
		if err := json.Unmarshal([]byte(data), &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "message_start":
			if ev.Message != nil {
				idleMergeUsage(&usage, ev.Message.Usage)
			}
		case "content_block_delta":
			if ev.Delta != nil && ev.Delta.Type == "text_delta" {
				if maxChars <= 0 || text.Len() < maxChars {
					text.WriteString(ev.Delta.Text)
				}
			}
		case "message_delta":
			if ev.Usage != nil {
				idleMergeUsage(&usage, *ev.Usage)
			}
			if sr := gjson.Get(data, "delta.stop_reason").String(); sr != "" {
				stop = sr
			}
		case "error":
			msg := gjson.Get(data, "error.message").String()
			return "", usage, stop, fmt.Errorf("upstream stream error: %s", sanitizeUpstreamErrorMessage(msg))
		}
	}
	if err := sc.Err(); err != nil {
		return "", usage, stop, err
	}
	return text.String(), usage, stop, nil
}

func idleMergeUsage(dst *ClaudeUsage, src apicompat.AnthropicUsage) {
	if src.InputTokens > dst.InputTokens {
		dst.InputTokens = src.InputTokens
	}
	if src.OutputTokens > dst.OutputTokens {
		dst.OutputTokens = src.OutputTokens
	}
	if src.CacheReadInputTokens > dst.CacheReadInputTokens {
		dst.CacheReadInputTokens = src.CacheReadInputTokens
	}
	if src.CacheCreationInputTokens > dst.CacheCreationInputTokens {
		dst.CacheCreationInputTokens = src.CacheCreationInputTokens
	}
}

// sendIdleCompactionRequest replays the snapshot request with a fresh token.
func (s *GatewayService) sendIdleCompactionRequest(ctx context.Context, snap *idleCompactionSnapshot, body []byte) (*http.Response, error) {
	token, tokenType, err := s.GetAccessToken(ctx, snap.account)
	if err != nil {
		return nil, fmt.Errorf("get access token: %w", err)
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, snap.url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header = snap.header.Clone()
	deleteHeaderAllForms(req.Header, "authorization")
	deleteHeaderAllForms(req.Header, "x-api-key")
	deleteHeaderAllForms(req.Header, "content-length")
	if tokenType == "oauth" {
		setHeaderRaw(req.Header, "authorization", "Bearer "+token)
	} else {
		setAnthropicAPIKeyAuthHeader(req.Header, snap.account, token, snap.account.GetBaseURL())
	}
	if getHeaderRaw(req.Header, "x-client-request-id") != "" {
		setHeaderRaw(req.Header, "x-client-request-id", uuid.NewString())
	}
	var profile *tlsfingerprint.Profile
	if s.tlsFPProfileService != nil {
		profile = s.tlsFPProfileService.ResolveTLSProfile(snap.account)
	}
	return s.httpUpstream.DoWithTLS(req, snap.proxyURL, snap.account.ID, snap.account.Concurrency, profile)
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

type idleCompactionState struct {
	Version int                     `json:"version"`
	Records []*idleCompactionRecord `json:"records"`
}

func (ic *idleCompactor) loadState() {
	if ic.cfg.StatePath == "" {
		return
	}
	raw, err := os.ReadFile(ic.cfg.StatePath)
	if err != nil {
		return
	}
	var st idleCompactionState
	if err := json.Unmarshal(raw, &st); err != nil || st.Version != idleCompactionStateVersion {
		logger.L().Warn("idle_compaction.state_ignored", zap.String("path", ic.cfg.StatePath))
		return
	}
	ic.mu.Lock()
	for _, r := range st.Records {
		if r != nil && r.Key != "" && r.Boundary > 0 && r.PrefixHash != "" && r.Summary != "" {
			ic.records[r.Key] = r
		}
	}
	ic.pruneRecordsLocked()
	ic.mu.Unlock()
}

func (ic *idleCompactor) saveState() {
	if ic.cfg.StatePath == "" {
		return
	}
	ic.mu.Lock()
	st := idleCompactionState{Version: idleCompactionStateVersion}
	for _, r := range ic.records {
		cp := *r
		st.Records = append(st.Records, &cp)
	}
	ic.mu.Unlock()
	sort.Slice(st.Records, func(i, j int) bool { return st.Records[i].CreatedAt.Before(st.Records[j].CreatedAt) })
	raw, err := json.Marshal(st)
	if err != nil {
		return
	}
	ic.saveMu.Lock()
	defer ic.saveMu.Unlock()
	if err := os.MkdirAll(filepath.Dir(ic.cfg.StatePath), 0o700); err != nil {
		logger.L().Warn("idle_compaction.state_save_failed", zap.Error(err))
		return
	}
	tmp := ic.cfg.StatePath + ".tmp"
	if err := os.WriteFile(tmp, raw, 0o600); err != nil {
		logger.L().Warn("idle_compaction.state_save_failed", zap.Error(err))
		return
	}
	if err := os.Rename(tmp, ic.cfg.StatePath); err != nil {
		logger.L().Warn("idle_compaction.state_save_failed", zap.Error(err))
	}
}

func (ic *idleCompactor) pruneRecordsLocked() {
	cutoff := ic.now().Add(-ic.cfg.RecordMaxAge)
	for k, r := range ic.records {
		if ic.cfg.RecordMaxAge > 0 && r.CreatedAt.Before(cutoff) {
			delete(ic.records, k)
		}
	}
	if ic.cfg.MaxRecords <= 0 || len(ic.records) <= ic.cfg.MaxRecords {
		return
	}
	all := make([]*idleCompactionRecord, 0, len(ic.records))
	for _, r := range ic.records {
		all = append(all, r)
	}
	sort.Slice(all, func(i, j int) bool { return all[i].CreatedAt.Before(all[j].CreatedAt) })
	for _, r := range all[:len(all)-ic.cfg.MaxRecords] {
		delete(ic.records, r.Key)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// idleDetectCacheTTL returns the TTL of the last ephemeral breakpoint in
// Anthropic prefix order (tools, system, messages). 0 = no breakpoint.
func idleDetectCacheTTL(body []byte) time.Duration {
	var ttl time.Duration
	visit := func(cc gjson.Result) {
		if !cc.Exists() || cc.Get("type").String() != "ephemeral" {
			return
		}
		if cc.Get("ttl").String() == "1h" {
			ttl = time.Hour
		} else {
			ttl = 5 * time.Minute
		}
	}
	gjson.GetBytes(body, "tools").ForEach(func(_, v gjson.Result) bool {
		visit(v.Get("cache_control"))
		return true
	})
	if sys := gjson.GetBytes(body, "system"); sys.IsArray() {
		sys.ForEach(func(_, v gjson.Result) bool {
			visit(v.Get("cache_control"))
			return true
		})
	}
	gjson.GetBytes(body, "messages").ForEach(func(_, m gjson.Result) bool {
		if content := m.Get("content"); content.IsArray() {
			content.ForEach(func(_, b gjson.Result) bool {
				visit(b.Get("cache_control"))
				return true
			})
		}
		return true
	})
	visit(gjson.GetBytes(body, "cache_control"))
	return ttl
}

// idleCompactionKey identifies a conversation: client credential + model +
// first message. The credential is hashed, never stored.
func idleCompactionKey(h http.Header, model string, first apicompat.AnthropicMessage) string {
	cred := ""
	if h != nil {
		for _, name := range []string{"Authorization", "X-Api-Key", "X-Goog-Api-Key"} {
			if v := strings.TrimSpace(h.Get(name)); v != "" {
				cred = v
				break
			}
		}
	}
	sum := sha256.New()
	_, _ = io.WriteString(sum, "idle-v1\x00"+cred+"\x00"+model+"\x00"+first.Role+"\x00")
	_, _ = sum.Write(first.Content)
	return hex.EncodeToString(sum.Sum(nil))
}

func idleChainHash(prev string, m apicompat.AnthropicMessage) string {
	sum := sha256.New()
	_, _ = io.WriteString(sum, prev+"\x00"+m.Role+"\x00")
	_, _ = sum.Write(m.Content)
	return hex.EncodeToString(sum.Sum(nil))
}

// idleCleanUserMessage reports whether m is a user turn that can start the
// verbatim tail: role=user, has text, and carries no tool_result blocks.
func idleCleanUserMessage(m apicompat.AnthropicMessage) (bool, string) {
	if m.Role != "user" || len(m.Content) == 0 {
		return false, ""
	}
	content := gjson.ParseBytes(m.Content)
	text := ""
	switch {
	case content.Type == gjson.String:
		text = content.String()
	case content.IsArray():
		clean := true
		content.ForEach(func(_, b gjson.Result) bool {
			switch b.Get("type").String() {
			case "tool_result":
				clean = false
				return false
			case "text":
				if text == "" {
					text = b.Get("text").String()
				}
			}
			return true
		})
		if !clean {
			return false, ""
		}
	default:
		return false, ""
	}
	text = strings.Join(strings.Fields(text), " ")
	if text == "" {
		return false, ""
	}
	return true, idleExcerpt(text, 120)
}

func idleExcerpt(s string, maxRunes int) string {
	if utf8.RuneCountInString(s) <= maxRunes {
		return s
	}
	r := []rune(s)
	return string(r[:maxRunes]) + "…"
}
