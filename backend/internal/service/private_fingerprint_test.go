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

	// 1. System migration uses <context_configuration> and "Configuration loaded."
	body := []byte(`{"model":"claude-opus-5-5","system":"Custom system instructions","messages":[{"role":"user","content":"first question"}]}`)
	rewritten := rewriteSystemForNonClaudeCode(body, "Custom system instructions")
	require.Equal(t, "<context_configuration>\nCustom system instructions\n</context_configuration>",
		gjson.GetBytes(rewritten, "messages.0.content.0.text").String())
	require.Equal(t, "Configuration loaded.",
		gjson.GetBytes(rewritten, "messages.1.content.0.text").String())
	require.NotContains(t, string(rewritten), "[System Instructions]")
	require.NotContains(t, string(rewritten), "Understood. I will follow these instructions.")

	// 2. Dynamic tool rewrite activates at > 4 tools with custom prefix pool & hex suffix
	tools := []string{"bash_exec", "file_read", "file_write", "web_browse", "subagent_task"}
	m := buildDynamicToolMap(tools)
	require.NotNil(t, m)
	for orig, fake := range m {
		require.NotEqual(t, orig, fake)
		require.NotRegexp(t, `\d{2}$`, fake, "must not end with two-digit index")
		require.Contains(t, fake, "_")
	}

	// 3. TLS fingerprint enabled by default for Anthropic OAuth
	acct := &Account{ID: 42, Platform: PlatformAnthropic, Type: AccountTypeOAuth}
	require.True(t, acct.IsTLSFingerprintEnabled())
	var nilTLSSvc *TLSFingerprintProfileService
	require.NotNil(t, nilTLSSvc.ResolveTLSProfile(acct))

	//Explicit false disables it
	acctDisabled := &Account{
		ID: 43, Platform: PlatformAnthropic, Type: AccountTypeOAuth,
		Extra: map[string]any{"enable_tls_fingerprint": false},
	}
	require.False(t, acctDisabled.IsTLSFingerprintEnabled())

	// 4. Session seed skips migrated <context_configuration> block so distinct conversations get distinct session IDs
	svc := &GatewayService{cfg: &config.Config{}}
	fp := &Fingerprint{ClientID: strings.Repeat("a", 64), UserAgent: claude.DefaultUserAgent()}
	b1 := rewriteSystemForNonClaudeCode([]byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"conv A opener"}]}`), "Same System")
	b2 := rewriteSystemForNonClaudeCode([]byte(`{"model":"claude-opus-5-5","messages":[{"role":"user","content":"conv B opener"}]}`), "Same System")
	uid1 := svc.buildOAuthMetadataUserIDFromBody(context.Background(), acct, fp, b1)
	uid2 := svc.buildOAuthMetadataUserIDFromBody(context.Background(), acct, fp, b2)
	require.NotEmpty(t, uid1)
	require.NotEqual(t, uid1, uid2, "conversations with identical system prompt but different user opener must not share session_id")

	// 5. Mimic headers include updated Stainless versions, accept-language, sec-fetch-mode, and X-Claude-Code-Session-Id
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
}
