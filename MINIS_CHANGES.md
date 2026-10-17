# Sub2API · Minis 修改版

本仓库 fork 自 [Wei-Shaw/sub2api](https://github.com/Wei-Shaw/sub2api)，基于上游固定提交
[`01842576`](https://github.com/Wei-Shaw/sub2api/commit/01842576b3d5c1e2545b5a99ad555d36e28ed306)（v0.2.16）。
改动集中在 **OpenAI Chat Completions → Anthropic** 这条转换链路。修改版用于给 Minis 这类 OpenAI 兼容客户端接入
Claude Opus 5.5 时，修掉缓存、思维链、空回复和流式传输方面的问题。

修改都在分支 `minis` 上，所有改动可以通过与 `01842576` 的 diff 完整审阅。上游许可证（LGPL-3.0）保持不变。

## 改动一览

| # | 问题 | 修改 | 开关（环境变量） |
|---|---|---|---|
| 1 | Opus 5.5 不返回思考内容 | 自适应思考请求加 `thinking.display=summarized`，客户端能看到思考摘要 | — |
| 2 | Chat 路径的缓存只有 5 分钟 | 管理后台的 1h 缓存 TTL 设置在 Chat 路径上也生效（原来只在 `/v1/messages` 生效） | `SUB2API_CHAT_CACHE_TTL_1H`（默认开；仍需管理后台开启 1h TTL） |
| 3 | 长对话停顿后要整段重建缓存 | **空闲压缩**：大会话空闲后，在后台把较早的历史压成摘要；下一次请求用摘要替换这段前缀 | `SUB2API_IDLE_COMPACTION_ENABLED`（默认关） |
| 4 | 思维链标记 `<thinking>` 出现在正文里 | 删掉 assistant 历史里的 `reasoning_content`，并去掉开头残留的 `<thinking>…</thinking>` 块，不再拼成正文发给上游 | — |
| 5 | `400 assistant message prefill` | 发送前去掉末尾的 assistant 消息（Claude 5.5 不支持 prefill） | — |
| 6 | 被上游安全策略拦截后回复无声中断 | `stop_reason=refusal` 不再伪装成空的正常结束，改为返回一条可见提示（附拦截类别） | — |
| 7 | 工具参数要等全部生成完才一次性到达 | 给自定义工具加 `eager_input_streaming: true`，参数边生成边推送 | `SUB2API_CHAT_EAGER_TOOL_STREAMING`（默认开） |

### 实测效果（真实请求）

- **缓存**：间隔 7.3 分钟发送同一个 7.3k token 的请求，原版缓存全部失效要重建，修改版全部命中。
- **工具参数流式**：参数传输到 25% / 50% / 75% 时间点时，原版只收到约 5% 的参数（其余压到最后一刻），修改版分别收到 18% / 51% / 76%。
- **prefill / 空 user 结尾**：原版返回 400，修改版正常回答。
- **refusal**：原版返回空内容、`finish_reason=stop`，修改版返回
  `[Sub2API] 本次回复被 Anthropic 上游安全策略拦截（reasoning_extraction）…`。

## 涉及文件

新增：
- `backend/internal/service/idle_compaction.go` / `_test.go`：空闲压缩
- `backend/internal/service/chat_sanitize.go` / `_test.go`：历史清理、prefill 修剪、refusal 提示
- `backend/internal/service/chat_eager_tools.go` / `_test.go`：工具参数实时流式
- `backend/internal/pkg/apicompat/thinking_display_local_test.go`

修改：
- `backend/internal/service/gateway_forward_as_chat_completions.go`：在 Chat 转换链路上接入以上功能
- `backend/internal/pkg/apicompat/types.go`、`responses_to_anthropic_request.go`：`thinking.display` 字段

## 空闲压缩的参数

| 变量 | 默认 | 说明 |
|---|---|---|
| `SUB2API_IDLE_COMPACTION_ENABLED` | `false` | 总开关 |
| `SUB2API_IDLE_COMPACTION_MIN_TOKENS` | `200000` | 上下文达到这个规模的会话才会压缩 |
| `SUB2API_IDLE_COMPACTION_KEEP_TOKENS` | `40000` | 最新的这部分上下文原样保留 |
| `SUB2API_IDLE_COMPACTION_AFTER` | 由缓存 TTL 推算 | 空闲多久后开始压缩（如 `54m`） |
| `SUB2API_IDLE_COMPACTION_ALLOW_5M` | `false` | 是否也压缩只有 5m 缓存的会话 |

压缩会把较早的细节换成摘要，属于有损操作，所以默认关闭。

## 构建

与上游相同：

```bash
cd frontend && pnpm install --frozen-lockfile && pnpm run build
cd ../backend && CGO_ENABLED=0 go build -tags embed -o sub2api ./cmd/server
```

测试：

```bash
cd backend
go test ./internal/service -run 'EagerTool|Sanitize|TrimTrailing|RefusalBecomes|IdleCompaction|ChatCompletions|CacheTTL'
go test ./internal/pkg/apicompat
```

## 已知限制

- 删掉历史里的思考内容后，模型看不到以前轮次的思考摘要。Chat Completions 协议里放不了思考签名，所以这些内容本来也不能作为真正的思考回放，删掉对记忆的影响很小。
- 思考摘要是上游分批压缩后推送的，思考很短时会和工具调用几乎同时到达。这一点网关改不了。
- 工具参数实时推送后，如果输出被截断（比如撞上 `max_tokens`），客户端可能收到不完整的 JSON。

## 许可证

LGPL-3.0，与上游一致，见 [LICENSE](LICENSE)。
