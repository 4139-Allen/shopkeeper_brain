# 查询流程（query_process）完整链路文档

> LangGraph 构建的 RAG 查询流水线：从用户原始问题出发，确认商品名后并行三路检索（向量 / HyDE / Web MCP），经 RRF 融合与 Rerank 精排，最终 LLM 生成答案并写入会话历史。

---

## 1. 总览

```
输入: original_query + session_id + task_id (+ is_stream)
输出: state['answer'](非流式落任务结果 / 流式 SSE 推送) + Mongo 会话历史
```

主图入口：`main_grapg.py` → `create_query_graph()` / 全局实例 `query_app`

> ⚠️ 文件名 `main_grapg.py` 为历史拼写（grapg），建议重命名为 `main_graph.py` 与导入流程一致。

---

## 2. 流程图

```mermaid
graph TB
    START --> item_name_confirm[item_name_confirm<br/>商品名确认+查询改写]
    item_name_confirm -->|route: 已有answer(澄清/致歉)| answer_output
    item_name_confirm -->|route: 无answer| multi_search(multi_search<br/>虚拟分发节点)
    multi_search --> vector_search[vector_search<br/>向量路检索]
    multi_search --> search_embedding_hyde[search_embedding_hyde<br/>HyDE路检索]
    multi_search --> web_search_mcp[web_search_mcp<br/>联网搜索路]
    vector_search --> join(join<br/>虚拟汇合节点)
    search_embedding_hyde --> join
    web_search_mcp --> join
    join --> rrf[rrf<br/>本地两路融合排序]
    rrf --> rerank[rerank<br/>多源精排+断崖截断]
    rerank --> answer_output[answer_output<br/>LLM生成答案+写历史]
    answer_output --> END_((END))
```

注册 key 与节点类对应关系：

| 图注册 key | 节点类 / 实现 | 文件 |
|---|---|---|
| `item_name_confirm` | `ItemNameConfirmNode` | `nodes/item_name_confirm.py` |
| `multi_search` | `lambda x: x`（虚拟分发） | — |
| `vector_search` | `VectorSearchNode` | `nodes/vector_search.py` |
| `search_embedding_hyde` | `HyDeSearchNode` | `nodes/hyde_search.py` |
| `web_search_mcp` | `WebSearchMcpNode` | `nodes/web_search_mcp.py` |
| `join` | `lambda x: {}`（虚拟汇合） | — |
| `rrf` | `RrfNode` | `nodes/rrf.py` |
| `rerank` | `RerankNode` | `nodes/rerank.py` |
| `answer_output` | `AnswerOutputNode` | `nodes/answer_output.py` |

条件路由：`route_after_item_confirm(state)` —— `state` 中已有 `answer`（商品歧义/识别失败时写入的澄清、致歉文案）→ 直接跳到 `answer_output` 短路输出；否则进入三路并行检索。

---

## 3. 节点逐个说明（完整链路）

### 3.1 item_name_confirm —— 商品名确认节点

```
上游传入的 original_query + session_id
        ↓
Mongo 拉取最近会话(get_recent_messages, limit=10)并拼成 history_text
        ↓
LLM 结合历史从问题中提取商品名 + 改写查询(失败降级为原查询)
        ↓
商品名 BGE-M3 混合向量化 → milvus item_name 集合混合检索(权重 0.5/0.5, 归一化)
        ↓
评分对齐: ≥0.7 且同名/唯一高分 → confirmed; 多候选或 0.6~0.7 → options; 其余丢弃
        ↓
分差过滤(多 confirmed 时剔除与最高分差 >0.15 的误判)
        ↓
state 更新 confirmed 商品名与 rewritten_query;歧义→澄清 answer;无识别→致歉 answer → 条件路由
```

- **读**：`original_query`、`session_id`
- **写**：`item_names`、`rewritten_query`、`history`；歧义/失败时写 `answer`（触发短路）
- 关键内部组件：
  - `ItemNameExtractor`：LLM 输出 JSON `{item_names, rewritten_query}`，清洗 ``` 代码块后反序列化，异常时返回默认值（`item_names=[]`、`rewritten_query=原查询`）。
  - `ItemNameAligner`：match（向量库查询）→ align（阈值对齐）→ filter（分差过滤）三步。

### 3.2 vector_search —— 向量路检索

```
确认后的 rewritten_query + item_names
        ↓
参数校验(缺失抛 StateFieldError)
        ↓
BGE-M3 encode_queries(is_query=True) 混合向量化
        ↓
构建 item_name in [...] 过滤表达式
        ↓
milvus chunks 集合混合检索 top5 → output_fields: chunk_id/content/item_name/title
        ↓
state['embedding_chunks'] = 检索结果 → join
```

### 3.3 search_embedding_hyde —— HyDE 路检索

> HyDE = Hypothetical Document Embeddings：**LLM 用在检索之前**（查询扩展），不是检索后再加工文档。

```
确认后的 rewritten_query + item_names
        ↓
LLM 扮演该商品技术文档专家,生成"假设性答案文档"(HYDE_USER_PROMPT_TEMPLATE)
        ↓
拼接 f"{查询}\n{假设文档}" → BGE-M3 混合向量化
        ↓
构建 item_name 过滤表达式 → milvus chunks 集合混合检索 top5
        ↓
state['hyde_embedding_chunks'] = 检索结果 → join
```

- 原理：假想文档在文体/长度上更接近真实手册段落，向量空间中与答案 chunk 距离更近，可召回直接嵌 query 召回不到的内容。
- 与 vector_search 用**同一个 chunks 集合**，仅嵌入载体不同。

### 3.4 web_search_mcp —— 联网搜索路

```
确认后的 rewritten_query + item_names
        ↓
参数校验(缺失抛 StateFieldError)
        ↓
MCP Streamable HTTP 连接百炼 WebSearch 服务(MCP_DASHSCOPE_BASE_URL 必须填完整 /mcp 端点)
        ↓
调用 bailian_web_search 工具联网检索(count=3)
        ↓
解析 TextContent 中 JSON 的 pages → 提取 title/url/snippet
        ↓
state['web_search_docs'] = 搜索结果 → join(交给 rerank 参与精排,不参与 RRF)
```

### 3.5 rrf —— RRF 融合排序节点

```
上游两路结果 embedding_chunks + hyde_embedding_chunks
        ↓
各路格式规整(_normalize_input 取 entity 内层文档,过滤无效项)
        ↓
按 RRF 公式累加得分: score += weight / (k + rank),k=rrf_k(60),两路权重均 1.0
        ↓
按分数降序排序,截取 top_k=rrf_max_results(10)
        ↓
state['rrf_chunks'] = 融合后的文档列表 → rerank
```

- 只融合**本地两路**；web 结果绕过 RRF 直接到 rerank 汇合。

### 3.6 rerank —— 重排序节点

```
上游 rrf_chunks(source=local) + web_search_docs(source=web) + 查询
        ↓
合并多源文档并统一格式 {content,title,chunk_id,url,source}
        ↓
BGE-M3 Reranker 对 (query, content) 逐对 compute_score(原始分数 -∞~+∞)
        ↓
sigmoid 归一化到 (0,1) → 按分数降序排序
        ↓
断崖检测: 扫描相邻分差,gap≥rerank_gap_abs(0.15) 的最大落差处截断;保底 rerank_min_top_k(3)
        ↓
state['reranked_docs'] = 精排文档 → answer_output
```

- 单文档防护：Reranker 单条输入可能返回 float，代码统一封装成 list。
- 重排模型失败降级：所有文档 `score=None` 原样返回（断崖检测会跳过 None 分差）。

### 3.7 answer_output —— 答案输出节点（终点）

```
上游 reranked_docs + 查询 + history (+ 可能已有的澄清/致歉 answer)
        ↓
已有 answer(短路分支写入) → 直接落任务结果,不再调 LLM
        ↓
无 answer → 按字符预算(max_context_chars=12000)组装:
  文档上下文([序号][source][chunk_id/url][title][score] + content)
  + 历史对话(用户/组手) + item_names + 问题 → ANSWER_PROMPT
        ↓
LLM 生成答案:is_stream=True 流式逐 token SSE 推送(DELTA) / False 一次性 invoke
        ↓
用户问题与助手回答写入 Mongo 历史(save_chat_message)
        ↓
state['answer'] + state['prompt'] 更新;流式模式最后推 FINAL 事件 → END
```

- **读**：`reranked_docs`、`rewritten_query/original_query`、`item_names`、`history`、`task_id`、`is_stream`、可能的 `answer`
- **写**：`answer`、`prompt`；任务结果（`set_task_result`）、SSE 事件（`push_sse_event`）、Mongo `chat_message` 集合

---

## 4. 状态字段（`state.py` → `QueryGraphState`，TypedDict total=False）

| 字段 | 类型 | 产生节点 | 说明 |
|---|---|---|---|
| `session_id` / `task_id` / `message_id` | str | 外部 | 会话/任务/消息标识 |
| `original_query` | str | 外部 | 用户原始问题 |
| `item_names` | list[str] | item_name_confirm | 确认后的商品名（=confirmed） |
| `rewritten_query` | str | item_name_confirm | LLM 改写后的检索查询 |
| `history` | list | item_name_confirm | Mongo 最近会话（原始消息 dict） |
| `embedding_chunks` | list | vector_search | 向量路混合检索结果 |
| `hyde_embedding_chunks` | list | search_embedding_hyde | HyDE 路检索结果 |
| `web_search_docs` | list | web_search_mcp | 联网搜索结果 {title,url,snippet} |
| `rrf_chunks` | list | rrf | 本地两路融合结果（含 chunk entity） |
| `reranked_docs` | list | rerank | 精排后统一格式文档（含 score/source） |
| `prompt` | str | answer_output | 最终提示词（留痕调试） |
| `answer` | str | item_name_confirm(短路) / answer_output | 最终答案 |
| `is_stream` | bool | 外部 | 是否流式 SSE 输出 |

---

## 5. 依赖的存储与服务

| 中间件 | 集合/表 | 用途 | 访问工具 |
|---|---|---|---|
| Milvus | `item_name_collection` | 商品名混合检索（确认节点） | `utils/milvus_util.py` |
| Milvus | `chunks_collection` | 文档块混合检索（vector/hyDE 两路） | 同上 |
| MongoDB | `chat_message` | 会话历史读/写 | `utils/mongo_history_util.py` |
| 百炼 LLM | DashScope 兼容模式 | 商品名提取/HyDE 假设文档/答案生成 | `utils/client/ai_clients.py` |
| BGE-M3 | 本地模型 | 查询/假想文档混合向量化 | 同上 |
| BGE-M3 Reranker | 本地模型 | (query,doc) 相关性打分 | 同上 |
| 百炼 WebSearch MCP | `MCP_DASHSCOPE_BASE_URL` | 联网搜索 | `agents.mcp.MCPServerStreamableHttp` |
| SSE 通道 | — | 流式推送 DELTA/FINAL 事件 | `utils/sse_util.py` |
| 任务存储 | — | 非流式答案落结果 | `utils/task_util.py` |

混合检索实现：`create_hybrid_search_requests`（dense ANN + sparse ANN）+ `execute_hybrid_search_query`（RRFRanker/加权融合，`norm_score` 可选归一化）。

---

## 6. 关键配置（`config.py` → `QueryConfig`，均支持 .env 覆盖）

| 配置 | 默认 | 作用 |
|---|---|---|
| `item_name_high_confidence` / `item_name_mid_confidence` | 0.7 / 0.6 | 商品名对齐阈值 |
| `item_name_max_options` | 5 | 歧义候选上限 |
| `item_name_dense_weight` / `item_name_sparse_weight` | 0.5 / 0.5 | 商品名混合检索权重 |
| `embedding_search_limit` / `hyde_search_limit` | 10 / 5 | 两路检索条数 |
| `rrf_k` / `rrf_max_results` | 60 / 10 | RRF 平滑参数与输出条数 |
| `rerank_max_top_k` / `rerank_min_top_k` | 10 / 3 | 精排截断上下界 |
| `rerank_gap_abs` | 0.15 | 断崖检测最小分差 |
| `max_context_chars` | 12000 | 答案提示词字符预算 |
| `dashscope_api_base` / `dashscope_api_key` / `default_model` / `item_model` | env | LLM |
| `milvus_url` / `chunks_collection` / `item_name_collection` | env | Milvus |
| `mcp_dashscope_base_url` | env `MCP_DASHSCOPE_BASE_URL` | 百炼 WebSearch MCP **完整端点**（如 `https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp`） |

---

## 7. 运行方式

```python
from knowledge.processor.query_process.main_grapg import query_app
from knowledge.processor.query_process.state import create_default_state

init_state = create_default_state(
    task_id="task_001",
    session_id="sess_001",
    original_query="RS-12 数字万用表怎么测试电阻？",
    is_stream=False,
)
final_state = query_app.invoke(init_state)
print(final_state.get("answer"))
```

- 直接运行 `main_grapg.py` 的 `__main__` 跑全链路。
- 各节点均可独立运行自测（每个节点文件底部有 `if __name__ == "__main__"` 测试入口与 mock state）。

---

## 8. 已知问题与注意事项

1. **`update_message_item_names` 仍是 TODO 空实现**（`utils/mongo_history_util.py`），商品名确认后的历史回填实际不生效。
3. **注册 key 与类 `name` 不一致**：`search_embedding_hyde` vs `HyDeSearchNode.name="hyde_search_node"`，stream 调试日志会混淆。
4. **MCP URL 必须完整**：`MCP_DASHSCOPE_BASE_URL` 缺失/只填 base 会报 `Unexpected content type`（空）或 `Not Found`(404)，需填百炼控制台给出的 `/mcp` 全路径。
5. **商品名写法对齐**：导入时入库的 `item_name` 若带品类后缀（如"××计算机"），而 LLM 从用户问题提取的是短名（"××L420x"），同名精确匹配会失败、落入 options 要用户澄清——导入提示词应避免品类后缀。
