# shopkeeper_brain — 店主知识库大脑

> 面向店主场景的 RAG 知识系统：把产品说明书类 PDF **导入**为可检索的知识库，再基于混合检索为自然语言提问**生成答案**。
> 全流程由 LangGraph 编排，AI 能力对接阿里云百炼（Qwen / BGE / Rerank / MCP 联网搜索），数据侧依赖 Milvus / MongoDB / MinIO。

---

## 1. 技术栈

| 领域 | 选型 | 说明 |
|---|---|---|
| 工作流编排 | LangGraph `1.2.12` | StateGraph 双链路：导入 7 节点串行、查询多路并行 |
| Web 框架 | FastAPI `0.143.0` + Uvicorn | 导入服务(8000) 与 查询服务(8001) 独立部署 |
| PDF 解析 | MinerU `2.1.1`（pipeline 后端） | 子进程调用，PDF → Markdown + 抽取图片 |
| LLM / VLM | langchain-openai `1.1.9`(`ChatOpenAI`, qwen) + openai SDK(qwen-vl 图片摘要) | 百炼 DashScope OpenAI 兼容模式 |
| 向量化 | BGE-M3（**pymilvus `3.0.2` 的 `BGEM3EmbeddingFunction`**，本地） | 1024 维稠密 + 稀疏双向量，CPU、离线 |
| 重排序 | **FlagEmbedding `1.3.5` 的 `FlagReranker`**（BGE-Reranker-Large） | sigmoid 归一 + 断崖截断 |
| 向量库 | Milvus（pymilvus `3.0.2`） | `hybrid_search` + `WeightedRanker(0.5, 0.5, norm_score=True)` 一次混合检索 |
| 对话历史 | MongoDB（pymongo） | `chat_message` 集合按 session_id 存取 |
| 对象存储 | MinIO | 说明书图片原件上传，md 内替换为远程 URL |
| 联网搜索 | 百炼 WebSearch MCP（`mcp==1.30.0` + `openai-agents`） | streamable_http 协议，外部兜底信源 |
| 流式推送 | SSE（Server-Sent Events） | `sse_util` 内存 `queue.Queue` + 断线感知 |

> **依赖清单只有一处**：`knowledge/requirements.txt`（104 个包，版本全钉死，注释里标了 FlagEmbedding / mcp 两处版本红线）。
> 它既不在 `shopkeeper_brain/` 根目录，也不存在名为 `re.txt` 的副本——历史文档（含 `环境说明.md`）曾用 `re.txt` 指代依赖清单，但该文件从未提交、磁盘上也没有，相关引用已订正。

## 2. 分层架构与目录结构

```
D:\Python\Project\shopkeeper_brain\        # 包根（导入路径以 knowledge.* 起头，命令都从这里发起）
└── knowledge/                             # 全部业务代码（Git 仓库根）
    ├── .env                               # 环境变量(模型名/Milvus/Mongo/MinIO/MCP 等)，已被 .gitignore 忽略
    ├── requirements.txt                   # 依赖清单(104 包钉死，重建环境唯一依据)
    ├── README.md                          # 本文档
    ├── 环境说明.md                         # MinerU 环境搭建/版本红线/排错
    ├── 项目架构.mermaid                     # 分层依赖关系图(api→services→schema→processor→prompt/utils)
    ├── api/                               # 【API 层】FastAPI 路由
    │   ├── import_api.py                  #   导入服务 :8000(页面/upload/status)
    │   └── query_api.py                   #   查询服务 :8001(query/history/stream-SSE)
    ├── schema/                            # 【契约层】pydantic 请求/响应模型
    │   ├── upload_schema.py               #   UploadResponse（另含一份重复的 TaskStatusResponse）
    │   ├── task_schema.py                 #   TaskStatusResponse（import_api 实际引用这份）
    │   └── query_schema.py                #   QueryRequest/QueryResponse/StreamSubmitResponse
    ├── services/                          # 【服务层】API 与 LangGraph 的桥接
    │   ├── import_file_service.py         #   保存上传文件+传MinIO+后台跑导入图+状态流转
    │   └── query_service.py               #   执行查询图/历史读写/答案取回
    ├── core/                              # 【基础设施】依赖注入与路径
    │   ├── depend.py                      #   get_import_file_service / get_query_service(单例)
    │   └── paths.py                       #   包根 / 本地临时目录 / 前端静态目录
    ├── processor/                         # 【流程编排层】两条 LangGraph 链路
    │   ├── import_processor/              #   导入链路 → 详见该目录 README.md
    │   │   ├── main_graph.py              #     图组装/注册/条件路由 + __main__ 自测
    │   │   ├── state.py / base.py         #     状态定义 / BaseNode 基类(自动记录进度与耗时)
    │   │   ├── config.py / exceptions.py  #     切片与模型配置 / 异常体系
    │   │   └── nodes/                     #     entry→pdf_to_md→md_img→document_split→item_name_recognition→bge_embedding→import_milvus
    │   └── query_process/                 #   查询链路 → 详见该目录 README.md
    │       ├── main_graph.py              #     图组装(含短路路由/虚拟分发与汇合/三路并行)
    │       ├── state.py / base.py         #     状态定义 / BaseNode 基类(流式时推送 progress)
    │       ├── config.py / exceptions.py  #     rerank/rrf/检索等配置 / 异常体系
    │       └── nodes/                     #     item_name_confirm→(vector_search|search_embedding_hyde|web_search_mcp)→rrf→rerank→answer_output
    ├── prompt/                            # 【提示词层】LLM 模板集中管理
    │   ├── import_process_prompt.py       #   商品名提取
    │   └── query_prompt.py                #   商品名提取(含历史)/HyDE/最终答案模板
    ├── utils/                             # 【工具层】
    │   ├── task_util.py                   #   任务状态与节点进度(内存注册表,轮询数据源,节点名→中文映射)
    │   ├── sse_util.py                    #   SSE 队列/事件打包/流式生成器
    │   ├── embedding_util.py              #   BGE-M3 稠密+稀疏向量化(查询/文档两种前缀)
    │   ├── milvus_util.py                 #   Milvus 混合检索请求构建与执行、item_name 过滤表达式
    │   ├── mongo_history_util.py          #   对话历史读写(update_message_item_names 仍是 TODO)
    │   ├── markdown_util.py               #   MD 表格降维为自然语言
    │   └── client/                        #   客户端单例工厂(双重检查锁)
    │       ├── base.py                    #     BaseClientManager: _require_env / _get_or_create
    │       ├── ai_clients.py              #     OpenAI(VLM)/ChatOpenAI(LLM,json与text两种)/BGE-M3/Reranker
    │       └── storage_clients.py         #     Milvus/Mongo/MinIO
    ├── front/                             # 【前端层】原生 HTML
    │   ├── import.html                    #   上传页 + 进度轮询展示
    │   └── chat.html                      #   对话页 + SSE 流式渲染
    ├── tem_cache/                         # 解析临时缓存(已 gitignore)
    └── temp_data/                         # 运行时上传/解析文件(按日期/task_id 分层,已 gitignore)
        （processor/import_processor/temp_dir/ 为本地 PDF 语料，同样已 gitignore）
```

**依赖方向**：`front → api → schema/services → core → processor → utils/prompt/client`，上层不可被下层反向引用；`schema` 是内外边界，processor 的 TypedDict 状态不穿透到 api 层。

## 3. 核心链路（概要）

### 3.1 导入链路（`processor/import_processor/README.md` 有全量细节）

```
POST /upload (multipart)
  → 保存本地 + 传 MinIO（记 upload_file 节点）→ BackgroundTasks 提交 LangGraph
  entry(按后缀路由: pdf→pdf_to_md 分支 | md→直接 md_img | 其他→END)
  → pdf_to_md(MinerU 子进程) → md_img(VLM 摘要+图片上传 MinIO+md 内替换 URL)
  → document_split(标题层级切分 + 表格降维) → item_name_recognition(LLM 提取商品名+向量化入库)
  → bge_embedding_chunks(双向量化) → import_milvus(按 file_title 幂等回写 + 纯文本直插)
前端轮询 GET /status/{task_id} 查看 status / done_list / running_list / durations
```

### 3.2 查询链路（`processor/query_process/README.md` 有全量细节）

```
POST /query (is_stream=true 时先建 SSE 队列,后台跑图,返回 task_id)
  item_name_confirm(LLM 提取商品名+改写查询 → 向量匹配商品名: 确认/澄清/致歉,可短路)
  → multi_search(虚拟分发) 并行三路: vector_search | search_embedding_hyde | web_search_mcp
  → join(虚拟汇合) → rrf(仅本地两路,k=60) → rerank(断崖截断 min3/max10,本地+网页合并)
  → answer_output(长度预算+历史→LLM,SSE 推 delta/final,Mongo 双端落库)
前端 GET /stream/{task_id} 建立 SSE 长连接消费事件;非流式则同步等待后直接返回 answer
```

## 4. API 端览

### 导入服务 `api/import_api.py`（:8000）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 302 → `/import.html` |
| GET | `/import.html` | 上传页面(FileResponse) |
| POST | `/upload` | 接收 PDF/docx/md，秒回 `UploadResponse(message, task_id)`，导入进后台任务 |
| GET | `/status/{task_id}` | 任务状态：`status / done_list / running_list / durations` |

> 导入服务只通过 FileResponse 返回上传页，未挂载 `/front` 静态目录（挂载代码是 `pass` 占位）。

### 查询服务 `api/query_api.py`（:8001）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 302 → `/chat.html` |
| GET | `/chat.html` | 对话页面 |
| GET | `/front/*` | 静态目录挂载（`StaticFiles`） |
| POST | `/query` | 提问入口；`is_stream=true` 立即返回 `StreamSubmitResponse(task_id, session_id)`，否则同步返回 `QueryResponse(answer)` |
| GET | `/stream/{task_id}` | SSE 长连接：`progress` / `delta` / `final` 事件 |
| GET | `/history/{session_id}?limit=50` | 拉取对话历史（`limit` 默认 50） |
| DELETE | `/history/{session_id}` | 清空会话历史 |

## 5. 外部依赖中间件

| 中间件 | 地址(.env) | 用途 | 相关配置 |
|---|---|---|---|
| Milvus | `MILVUS_URL=http://192.168.20.101:19530` | `kb_chunks_v1`(切片双向量) + `kb_item_names_v1`(商品名) | `CHUNKS_COLLECTION` / `ITEM_NAME_COLLECTION` / `MILVUS_METRIC_TYPE` |
| MongoDB | `MONGO_URL=mongodb://192.168.20.101:27017` | `kb001.chat_message` 对话历史 | `MONGO_DB_NAME` |
| MinIO | `MINIO_ENDPOINT=192.168.20.101:9000` | 图片原件 + 原始文件归档 | `MINIO_BUCKET_NAME=knowledge-base-files` |
| DashScope | `QWEN_BASE_URL=.../compatible-mode/v1` | LLM/商品名/HyDE/答案生成/VLM 摘要 | `QWEN_API_KEY` / `LLM_DEFAULT_MODEL` / `VL_MODEL` / `ITEM_MODEL` |
| 百炼 MCP | `MCP_DASHSCOPE_BASE_URL=.../api/v1/mcps/WebSearch/mcp` | 联网搜索（注意必须是含 `/mcp` 的完整端点） | |
| 本地模型 | `HF_HOME=D:/ai_models/huggingface_cache` | MinerU 排版模型 + BGE-M3 + Reranker（CPU、离线） | `BGE_M3_PATH` / `BGE_RERANKER_LARGE` / `BGE_DEVICE=cpu` / `BGE_FP16=0` / `BGE_RERANKER_DEVICE` / `BGE_RERANKER_FP16` |

> `LLM_DEFAULT_MODEL`（模型名）由 `utils/client/ai_clients.py` 直接读取；`VL_MODEL` 供 `md_img` 图片摘要；`ITEM_MODEL` / `MODEL` 在 config 里有字段但当前链路未实际使用。

## 6. 快速开始

### 6.1 环境准备

```powershell
# 依赖以 knowledge/requirements.txt 全量钉死安装
# （transformers ≤4.54.0 与 FlagEmbedding==1.3.5、mcp 1.x 是红线，详见 knowledge/环境说明.md）
D:\Python\Project\shopkeeper_brain\knowledge\.venv\Scripts\python.exe -m pip install `
  -r D:\Python\Project\shopkeeper_brain\knowledge\requirements.txt

# 自检（期望输出 4.53.3 2.12.1）
D:\Python\Project\shopkeeper_brain\knowledge\.venv\Scripts\python.exe -c "import transformers, torch, mineru; print(transformers.__version__, torch.__version__)"
```

### 6.2 配置

编辑 `knowledge/.env`：百炼 API Key、Milvus/Mongo/MinIO 地址与桶、模型名（必须是 DashScope 兼容模式的**有效**名称）、集合名。
> 注意：各 config 通过 `load_dotenv()` 加载该文件（锚定 `knowledge/`），启动目录不影响加载位置。

### 6.3 启动

代码内部一律使用 `knowledge.*` 绝对包导入，所以**必须从包根 `D:\Python\Project\shopkeeper_brain` 以模块方式启动**（直接 `python knowledge/api/import_api.py` 会因 `sys.path` 只含脚本目录而报 `ModuleNotFoundError: No module named 'knowledge'`）：

```powershell
cd D:\Python\Project\shopkeeper_brain

# 两个服务分别启动(也可各自独立运行)
knowledge\.venv\Scripts\python.exe -m knowledge.api.import_api     # :8000 上传+进度
knowledge\.venv\Scripts\python.exe -m knowledge.api.query_api      # :8001 问答+SSE流式
```

浏览器打开 `http://localhost:8000` 上传说明书 → 等进度全绿；打开 `http://localhost:8001` 提问验证流式回答。

### 6.4 单节点调试（不起服务直接跑图）

```powershell
cd D:\Python\Project\shopkeeper_brain

knowledge\.venv\Scripts\python.exe -m knowledge.processor.import_processor.main_graph   # 单测导入图
knowledge\.venv\Scripts\python.exe -m knowledge.processor.query_process.main_graph     # 单测查询图(改 __main__ 里的示例问题)
knowledge\.venv\Scripts\python.exe -m knowledge.processor.import_processor.nodes.pdf_to_md  # 各节点自带 __main__
```

## 7. 关键机制速查

| 机制 | 位置 | 一句话说明 |
|---|---|---|
| 任务进度追踪 | `utils/task_util.py` | BaseNode.__call__ 自动 add_running/done/duration；内存 dict 按 task_id 隔离，重启即清零 |
| SSE 流式 | `utils/sse_util.py` | 生产者(节点/LLM) push 进 `queue.Queue` → 消费者 `sse_generator` 打包 SSE 包；收到 `final` 即断流并清理队列 |
| 双重检查锁单例 | `utils/client/base.py` | 所有客户端(get_milvus_client / get_llm_openai 等)进程内单例，懒创建、线程安全 |
| 商品名确认短路 | `query_process/main_graph.py` | item_name_confirm 已写 answer(澄清/致歉)则跳过检索直达 answer_output |
| 评分对齐 | `nodes/item_name_confirm.py` | ≥0.7 且同名精确/该路唯一命中→confirmed；≥0.7 多候选或 0.6~0.7→options(澄清)；无 mid→致歉；多 confirmed 再按 0.15 分差过滤 |
| RRF + 断崖截断 | `nodes/rrf.py` `nodes/rerank.py` | 本地两路 rank 融合(k=60,各权重 1.0) → reranker 打分 sigmoid 归一 → 相邻分差≥0.15 截断，保底 3 条、上限 10 条 |
| 幂等写入 | `nodes/item_name_recognition.py` `import_milvus.py` | 按 `file_title` 先 delete 后 insert，重复导入不产生脏数据 |
| 降级兜底 | 各检索节点/prompt | Milvus/LLM/MCP 异常不阻断图：空结果 + 日志 warning + 默认话术 |

## 8. 已知限制与待办

- **进度/队列为内存态**：`task_util`、`sse_util` 均为进程内字典——服务重启进度清零、SSE 队列丢失；多 worker 部署会导致轮询/流式打不中同一进程（未来可落 Redis）。
- **服务必须单 worker**：同上原因，uvicorn 不能开 `--workers > 1`。
- **历史回填未实现**：`utils/mongo_history_util.py:update_message_item_names` 仍是 `pass` 的 TODO（`item_name_confirm` 已正确地调用它，但回填不生效）。
- **检索路 limit 硬编码**：`vector_search.py` / `hyde_search.py` 都写死 `limit=5`，`QueryConfig.embedding_search_limit`(10) / `hyde_search_limit`(5) 未被子节点读取。
- **`bge_embedding.py` 异常分支有缺陷**：`encode_documents` 抛异常时 except 里只记日志后继续执行循环，会引用未定义的 `embedding_result`（UnboundLocalError）。
- **`answer_output.py` 分数格式串失效**：`f"[score={score}:.4f]"` 的 `:.4f` 未生效，会原样打印。
- **商品名带品类后缀**：LLM 提取的商品名（如"华为擎云 L420"）与库中含后缀的 name（"...计算机"）无法同名精确匹配，易多候选走澄清；治本方案是统一导入侧提取规则。
- **item_name 入库异常被吞**：`item_name_recognition._insert_milvus` 整段 try/except 只记日志；且向量化失败时 chunks 仍会回填 item_name，导致 chunks 携带 item_name 却未写入 item_name 集合。
- **切片无重叠**：`document_split` 用 `RecursiveCharacterTextSplitter(chunk_overlap=0)`，`ImportConfig.overlap_sentences` 未被使用。
- **VLM 批量失败无熔断**：模型名配置错误时会对每张图重复请求到速率限制（`md_img` 内滑动窗口限流，`requests_per_minute=10`），大批量图片会空耗数分钟（建议 fail-fast）。
- **前端与后端漂移**：`front/chat.html` 的 `poll()` 函数是死代码，且引用了查询服务从不返回的 `image_urls` / `error` 字段；`/status/{task_id}` 只存在于导入服务(8000)，查询服务(8001)无此路由。
- **契约层重复定义**：`schema/upload_schema.py` 与 `schema/task_schema.py` 各自定义了一份 `TaskStatusResponse`（import_api 引用的是 task_schema 那份）。
- **节点注册 key 与 `name` 属性不同**：查询图把 HyDE 节点注册为 `search_embedding_hyde`（`name="hyde_search_node"`）、导入图把入口注册为 `entry_node`（`name="entry"`）；`task_util` 按 `self.name` 映射中文进度名，故功能正常，仅可读性受影响。

## 9. 文档导航

| 文档 | 内容 |
|---|---|
| `knowledge/processor/import_processor/README.md` | 导入链路全量文档：逐节点职责、状态字段、Milvus schema、配置、排错 |
| `knowledge/processor/query_process/README.md` | 查询链路全量文档：短路/并行拓扑、评分规则、RRF/rerank 参数、SSE 事件约定 |
| `knowledge/环境说明.md` | MinerU 环境重建、版本红线（transformers ≤ 4.54.0）、排错速查 |
| `knowledge/项目架构.mermaid` | 分层依赖关系图 |

> ⚠️ `import_processor/README.md` 与 `query_process/README.md` 仍把主图文件写作 `main_grapg.py`（历史拼写），**实际文件名为 `main_graph.py`**（本文档与代码为准）。
>
> `环境说明.md` 中关于依赖清单（`re.txt` → `knowledge/requirements.txt`）与嵌入模型（`bge-small-zh-v1.5` → BGE-M3）的过期描述已订正。

## 10. 开发约定（摘要）

- LangGraph **注册即执行**：`add_node` 必须传可调用对象（`节点实例` 或 `lambda`），不能传协程函数引用；
- 节点统一经 `BaseNode.__call__` 包装：自动记录进度/耗时/日志，子节点只实现 `process`；
- state 字段读取用 `state.get(...)` 主动校验，缺失抛 `ValidationError` / `StateFieldError`；
- dict 取值对"键存在但值为 None"用 `or ""` 兜底，外部输入一律带默认值；
- 数据写入（Milvus 向量）必须幂等：先按业务键（`file_title`）删除再插入；
- 文件上传等写操作接口使用 POST；
- 一切内部导入以 `knowledge.*` 为根，脚本需从包根 `-m` 方式运行。
