package service

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/claude"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
	"github.com/tidwall/gjson"
)

func TestPrivateFingerprintCustomizations(t *testing.T) {
	gin.SetMode(gin.TestMode)

	// 1. System migration uses <context_configuration> and "Configuration loaded.",
	//    and cc_version suffix varies when underlying system instructions differ.
	bodyA := []byte(`{"model":"claude-opus-5-5","system":"Alpha instructions for project one","messages":[{"role":"user","content":"first question"}]}`)
	bodyB := []byte(`{"model":"claude-opus-5-5","system":"Zeta instructions for project two","messages":[{"role":"user","content":"first question"}]}`)
	rewrittenA := rewriteSystemForNonClaudeCode(bodyA, "Alpha instructions for project one")
	rewrittenB := rewriteSystemForNonClaudeCode(bodyB, "Zeta instructions for project two")
	require.Equal(t, "<context_configuration>\nAlpha instructions for project one\n</context_configuration>",
		gjson.GetBytes(rewrittenA, "messages.0.content.0.text").String())
	require.Equal(t, "Configuration loaded.",
		gjson.GetBytes(rewrittenA, "messages.1.content.0.text").String())
	require.NotContains(t, string(rewrittenA), "[System Instructions]")
	require.NotContains(t, string(rewrittenA), "Understood. I will follow these instructions.")
	fpA := computeClaudeCodeFingerprint(rewrittenA, "2.1.258")
	fpB := computeClaudeCodeFingerprint(rewrittenB, "2.1.258")
	require.NotEqual(t, fpA, fpB, "cc_version fingerprint must vary across different migrated system prompts")

	// 2. normalizeClaudeOAuthRequestBody does NOT inject empty tools:[], and defaults max_tokens to 64000
	normOut, _ := normalizeClaudeOAuthRequestBody([]byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"hi"}]}`), "claude-opus-5-5", claudeOAuthNormalizeOptions{})
	require.False(t, gjson.GetBytes(normOut, "tools").Exists(), "must not inject empty tools array when absent")
	require.Equal(t, int64(64000), gjson.GetBytes(normOut, "max_tokens").Int())

	// 3. Static & dynamic tool rewrites
	require.Equal(t, "wk_sess_list", sanitizeToolName("sessions_list", nil))
	require.Equal(t, "wk_ses_get", sanitizeToolName("session_get", nil))
	tools := []string{"bash_exec", "file_read", "file_write", "web_browse", "subagent_task"}
	m := buildDynamicToolMap(tools)
	require.NotNil(t, m)
	for orig, fake := range m {
		require.NotEqual(t, orig, fake)
		require.NotRegexp(t, `\d{2}$`, fake, "must not end with two-digit index")
		require.Contains(t, fake, "_")
	}

	// 4. TLS fingerprint enabled by default for Anthropic OAuth
	acct := &Account{ID: 42, Platform: PlatformAnthropic, Type: AccountTypeOAuth}
	require.True(t, acct.IsTLSFingerprintEnabled())
	var nilTLSSvc *TLSFingerprintProfileService
	require.NotNil(t, nilTLSSvc.ResolveTLSProfile(acct))

	acctDisabled := &Account{
		ID: 43, Platform: PlatformAnthropic, Type: AccountTypeOAuth,
		Extra: map[string]any{"enable_tls_fingerprint": false},
	}
	require.False(t, acctDisabled.IsTLSFingerprintEnabled())

	// 5. Session seed skips migrated <context_configuration> block so distinct conversations get distinct session IDs
	svc := &GatewayService{cfg: &config.Config{}}
	fp := &Fingerprint{ClientID: strings.Repeat("a", 64), UserAgent: claude.DefaultUserAgent()}
	b1 := rewriteSystemForNonClaudeCode([]byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"conv A opener"}]}`), "Same System")
	b2 := rewriteSystemForNonClaudeCode([]byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"conv B opener"}]}`), "Same System")
	uid1 := svc.buildOAuthMetadataUserIDFromBody(context.Background(), acct, fp, b1)
	uid2 := svc.buildOAuthMetadataUserIDFromBody(context.Background(), acct, fp, b2)
	require.NotEmpty(t, uid1)
	require.NotEqual(t, uid1, uid2, "conversations with identical system prompt but different user opener must not share session_id")

	// 6. Mimic headers preserve per-account Fingerprint (OS/Arch/RuntimeVersion/PackageVersion) instead of overwriting with static defaults
	b1WithMeta, ok := ensureClaudeOAuthMetadataUserID(b1, uid1)
	require.True(t, ok)
	c, _ := gin.CreateTestContext(httptest.NewRecorder())
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req, _, err := svc.buildUpstreamRequest(context.Background(), c, acct, b1WithMeta, "tok", "oauth", "claude-opus-5-5", true, true)
	require.NoError(t, err)
	require.Equal(t, "0.98.0", getHeaderRaw(req.Header, "X-Stainless-Package-Version"))
	require.Equal(t, "v24.11.0", getHeaderRaw(req.Header, "X-Stainless-Runtime-Version"))
	require.Equal(t, "*", getHeaderRaw(req.Header, "accept-language"))
	require.Equal(t, "cors", getHeaderRaw(req.Header, "sec-fetch-mode"))
	parsedUID := ParseMetadataUserID(uid1)
	require.NotNil(t, parsedUID)
	require.Equal(t, parsedUID.SessionID, getHeaderRaw(req.Header, "X-Claude-Code-Session-Id"))

	// Verify per-account fingerprint preservation when applied prior to applyClaudeCodeMimicHeaders
	reqCustom, err := http.NewRequest(http.MethodPost, "https://api.anthropic.com/v1/messages", nil)
	require.NoError(t, err)
	idSvc := &IdentityService{}
	idSvc.ApplyFingerprint(reqCustom, &Fingerprint{
		UserAgent:               claude.DefaultUserAgent(),
		StainlessLang:           "js",
		StainlessPackageVersion: "0.96.0",
		StainlessOS:             "MacOS",
		StainlessArch:           "x64",
		StainlessRuntime:        "node",
		StainlessRuntimeVersion: "v22.17.1",
	})
	applyClaudeCodeMimicHeaders(reqCustom, true, claude.DefaultUserAgent())
	require.Equal(t, "MacOS", getHeaderRaw(reqCustom.Header, "X-Stainless-OS"))
	require.Equal(t, "x64", getHeaderRaw(reqCustom.Header, "X-Stainless-Arch"))
	require.Equal(t, "v22.17.1", getHeaderRaw(reqCustom.Header, "X-Stainless-Runtime-Version"))
	require.Equal(t, "0.96.0", getHeaderRaw(reqCustom.Header, "X-Stainless-Package-Version"))
}
