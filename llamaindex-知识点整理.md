# LlamaIndex 官网使用说明 · 知识点整理

> 资料来源：LlamaIndex 官方文档站 `developers.llamaindex.ai`（原 docs.llamaindex.ai），Python 框架文档，整理日期 2026-10-26。
> 官网支持在任意文档 URL 后加 `index.md` 获取纯 Markdown 原文；另提供 `llms.txt` 和 `llms-full.txt` 全文索引。

---

## 目录

1. [LlamaIndex 是什么](#1-llamaindex-是什么)
2. [安装与环境设置](#2-安装与环境设置)
3. [核心概念总览](#3-核心概念总览)
4. [第一阶段：数据加载（Loading / Ingestion）](#4-第一阶段数据加载loading--ingestion)
5. [第二阶段：索引（Indexing）](#5-第二阶段索引indexing)
6. [第三阶段：存储（Storing）](#6-第三阶段存储storing)
7. [第四阶段：查询（Querying）](#7-第四阶段查询querying)
8. [第五阶段：评估与可观测性](#8-第五阶段评估与可观测性)
9. [Agent（智能体）](#9-agent智能体)
10. [Tools（工具）](#10-tools工具)
11. [Workflows（事件驱动工作流）](#11-workflows事件驱动工作流)
12. [LLM / Embedding 配置（Settings）](#12-llm--embedding-配置settings)
13. [结构化输出（Structured Output）](#13-结构化输出structured-output)
14. [进阶主题速览](#14-进阶主题速览)
15. [推荐学习路径](#15-推荐学习路径)

---

## 1. LlamaIndex 是什么

**定位**：一个简单、灵活的数据框架，用于**将你的私有数据接入 LLM 应用**。核心卖点是"上下文增强"（Context Augmentation）——让 LLM 能基于你的数据回答问题、执行任务。

**两大核心场景**：

| 场景 | 说明 |
|---|---|
| **RAG**（Retrieval-Augmented Generation，检索增强生成） | 把你的数据变成 LLM 可以查询的知识库（文档问答、聊天机器人） |
| **Agent**（智能体） | 让 LLM 成为推理引擎，能理解任务、分解步骤、调用工具并执行 |

**产品矩阵**：
- `llama-index`（开源 Python 框架，本文主角）
- `llama-index-workflows`（独立的事件驱动工作流库，已内置于 core）
- **LlamaCloud**：托管服务，包含 **LlamaParse**（文档解析，擅长 PDF/复杂版式）和 **LlamaCloudIndex**（托管 RAG 索引）
- TypeScript / JS 版本：`npm install @llama-index/core`

**一句话理解**：LangChain 偏"编排各种 LLM 调用链"，LlamaIndex 偏"数据接入层"——读取、切分、索引、检索你的数据，是做 RAG 的专用利器。

---

## 2. 安装与环境设置

### 2.1 安装

```bash
# 完整包（含 OpenAI 默认集成）
pip install llama-index

# 或按需安装（推荐，体积小）
pip install llama-index-core                      # 核心库
pip install llama-index-llms-openai               # OpenAI LLM 集成
pip install llama-index-embeddings-openai         # OpenAI Embedding
pip install llama-index-readers-file              # 文件 Reader
```

> **知识点**：LlamaIndex 从 v0.10 起采用"核心 + 插件"架构。所有集成（LLM、Embedding、向量库、Reader、后处理器）都是独立的 `llama-index-xxx` PyPI 包，安装哪块用哪块。

### 2.2 设置 API Key

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
```

### 2.3 全局配置 `Settings`（重点）

所有模块的默认 LLM 和 Embedding 模型通过 `Settings` 单例配置：

```python
from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

# 本地 Ollama LLM（无需 API Key）
Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)

# 本地 Embedding
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# 验证 LLM 可用
message = ChatMessage(role="user", content="写一首关于数据的小诗")
resp = Settings.llm.chat([message])
print(resp)
```

> **知识点**：不配置 `Settings.llm` 时默认用 OpenAI `gpt-4o-mini`；不配置 `Settings.embed_model` 时默认 `text-embedding-ada-002`。任何 OpenAI 兼容接口（DeepSeek、Qwen、GLM 等）都可以通过 `llama-index-llms-openai-like` 的 `OpenAILike` 类接入，只需改 `api_base` 和 `model`。

---

## 3. 核心概念总览

### 3.1 RAG 五阶段（官网主线）

官网将构建 LLM 应用的过程总结为 **5 个阶段**，整个文档体系就是围绕这 5 阶段组织的：

```
① Loading（加载）  →  ② Indexing（索引）  →  ③ Storing（存储）  →  ④ Querying（查询）  →  ⑤ Evaluating（评估）
```

### 3.2 核心对象速查表（官网 High-Level Concepts 页）

| 对象 | 作用 |
|---|---|
| **Document / Node** | 文档容器。`Document` 是一个整体文件；`Node` 是切分后的 chunk，带 `node_id`、前后邻居关系（`prev_node`/`next_node`）、score 等字段 |
| **Reader / Data Connector** | 从各类数据源（文件、Notion、Slack、网页……）读取数据，输出 `Document` 对象 |
| **Text Splitter** | 把 Document 切成 Node（chunk） |
| **Index** | 对 Node 建立的数据结构（最常用是向量索引），供检索 |
| **Retriever** | 给定查询，从索引中取相关上下文 |
| **Query Engine** | Retriever 之上包一层 LLM 合成：**检索 → 增强提示 → 生成回答** |
| **Chat Engine** | 有对话记忆的查询引擎，支持多轮 |
| **Agent** | LLM + 工具 + 记忆，能自主决定下一步动作（查询引擎只是它的工具之一） |
| **Tool** | Agent 可调用的函数，如搜索引擎、`QueryEngineTool`、自定义函数 |
| **Storage Context** | 管理三类存储：文档库、索引库、向量库 |
| **Response Synthesizer** | 把检索到的 Node 组装成 Prompt 交给 LLM 生成最终回答 |
| **Router** | 根据查询自动选择用哪个查询引擎/索引 |

### 3.3 五行代码版 RAG（官网招牌示例）

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()   # ① 加载
index = VectorStoreIndex.from_documents(documents)       # ② 索引
query_engine = index.as_query_engine()                   # ④ 查询引擎
response = query_engine.query("What did the author do growing up?")
print(response)                                          # 生成回答
```

---

## 4. 第一阶段：数据加载（Loading / Ingestion）

> 官网路径：`Getting Started → Loading Data (Ingestion)` 与 `Module Guides → Loading`

### 4.1 核心流程：`Ingestion` = Reader → Transformations → VectorStore

```
数据源 → [Data Reader] → Documents → [TextSplitter/Embedding 等变换] → Nodes → [向量库/文档库]
```

### 4.2 Reader（数据连接器）

最常用的是 `SimpleDirectoryReader`，支持 pdf、docx、csv、md、html、图片等几十种格式：

```python
from llama_index.core import SimpleDirectoryReader

# 读整个目录
documents = SimpleDirectoryReader("./data").load_data()

# 只读指定文件
documents = SimpleDirectoryReader(
    input_files=["./data/paul_graham_essay.txt"]
).load_data()

# 惰性读取（配合流水线）
reader = SimpleDirectoryReader("./data")
docs_iter = reader.iter_data()   # 每个文件一个 list[Document]
```

官网强调有 **数百种数据连接器**（LlamaHub）：Notion、Slack、Discord、Google Docs、S3、数据库……用法都是 `pip install llama-index-readers-xxx` 然后 `loader.load_data()`。

**自定义 Reader 例子**：

```python
from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.readers.base import BaseReader

class MyReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
        # extra_info 会作为 metadata 附加到 Document
        return [Document(text=text, metadata=extra_info or {})]
```

### 4.3 Document 与 Node

```python
from llama_index.core import Document
from llama_index.core.schema import TextNode

# Document：全量文档
doc = Document(
    text="这是文档正文……",
    metadata={"filename": "report.pdf", "author": "张三"},
)

# Node：切分后的最小检索单元
node = TextNode(
    text="这是一段文本块",
    id_="node-001",
    metadata={"page": 3},
)
```

### 4.4 文本切分（Node Parser / Text Splitter）

```python
from llama_index.core.node_parser import SentenceSplitter

# 按句子边界切分，避免句子被拦腰截断
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=20)
nodes = splitter.get_nodes_from_documents(documents)

print(nodes[0].text)               # 第一个 chunk 内容
print(nodes[0].metadata)           # 继承文档的 metadata
print(nodes[0].relationships)      # SOURCE/PREVIOUS/NEXT 关系
```

官网还提供 `TokenTextSplitter`（按 token 切）、`SentenceWindowNodeParser`（句子窗口）、`SemanticSplitterNodeParser`（语义切分）、`HierarchicalNodeParser`（层级切分，配合自动合并检索）等。

### 4.5 Metadata 与文档管理（重点）

```python
# 1) Document 上直接设置
doc = Document(text="……", metadata={"filename": "node_doc.md"})

# 2) 全局统一附加（注意：设置后加载的所有文档共享）
SimpleDirectoryReader.input_files = ["docs/node_101.md"]
# SimpleDirectoryReader 原生注入 filename、file_type 等默认元数据

# 3) 通过 docstore 追踪 ingestion 的文档
from llama_index.core import StorageContext
storage_context = StorageContext.from_defaults()
storage_context.docstore.add_documents(documents)
```

> **知识点**：metadata 会随 chunk 一起进入向量库并在检索时回传给 LLM，常用于**来源引用、按文件过滤检索**。

### 4.6 IngestionPipeline（生产级摄取流水线，重点）

官网将其列为"处理数据、按变换链拆分、可缓存、可并行"的标准方式：

```python
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.embeddings.openai import OpenAIEmbedding

pipeline = IngestionPipeline(
    transformations=[
        TokenTextSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(),          # 用 LLM 给每个 chunk 生成标题元数据
        QuestionsAnsweredExtractor(questions=3),  # 生成"该段能回答什么问题"元数据
        OpenAIEmbedding(),
    ],
    vector_store=my_vector_store,  # 可选：直接写入向量库
)

nodes = pipeline.run(documents=documents, show_progress=True)
```

**三大高级能力**（官网 IngestionPipeline 专页）：

1. **缓存**：`pipeline.persist("./pipeline_storage")` —— 变换结果落盘，重跑只处理新数据；
2. **Docstore 策略去重**：`docstore=SimpleDocumentStore()` + `docstore_strategy=DocstoreStrategy.UPSERTS`，配合 `pipeline.update_docstore()` 实现增量更新/删除；
3. **并行化**：`num_workers=4` 用多进程跑变换（文档多时提速明显）。

```python
# 增量摄取完整模式
from llama_index.core.ingestion import IngestionPipeline, DocstoreStrategy
from llama_index.core.storage.docstore import SimpleDocumentStore

pipeline = IngestionPipeline(
    transformations=[splitter, embedding],
    docstore=SimpleDocumentStore(),
    docstore_strategy=DocstoreStrategy.UPSERTS,
    vector_store=vector_store,
)
pipeline.update_docstore(documents)   # 记录哪些文档进过管道
nodes = pipeline.run(documents=documents)
```

---

## 5. 第二阶段：索引（Indexing）

> 官网路径：`Understanding → Indexing`、`Module Guides → Indexing → How Each Index Works`

**定义**：Index 是一种把数据组织起来便于 LLM 检索的数据结构。所有索引都要先有 Document/Node 再构建。

### 5.1 六种索引类型（官网专页"每种索引如何工作"）

| 索引类型 | 原理 | 适用场景 | 示例代码 |
|---|---|---|---|
| **VectorStoreIndex**（最常用） | 每个 Node 算 embedding，按**余弦相似度**取 top-k | 语义检索、大多数问答场景 | 见 3.3 |
| **SummaryIndex**（原 ListIndex） | 顺序遍历**所有** Node 交给 LLM | 需要"完整阅读"全文档的场景 | `SummaryIndex(nodes)` |
| **TreeIndex** | 把 Node 自底向上**汇总成树**，查询时按树下降 | 概括长文、分层摘要 | `TreeIndex(nodes)` |
| **KeywordTableIndex** | 从每个 Node 抽取关键词建**倒排表** | 精确关键词匹配场景 | `KeywordTableIndex(nodes)` |
| **DocumentSummaryIndex** | 为每个文档生成摘要并映射到其 Node | "先找文档再进文档"的两级检索 | `DocumentSummaryIndex.from_documents(docs)` |
| **KnowledgeGraphIndex / PropertyGraphIndex** | 从文本抽取三元组构建知识图谱 | 多跳推理、关系型问题 | `PropertyGraphIndex.from_documents(docs)` |

官网流程图要点：向量索引查询 = query 向量化 → 与全部 node 向量算相似度 → 取 top-k；摘要索引 = 按顺序喂给 LLM；树索引 = 逐层摘要后自顶向下查询。

### 5.2 从"已存在的向量库"直接构建

```python
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 已有向量的集合 → 直接建索引（免重新 embedding）
index = VectorStoreIndex.from_vector_store(vector_store)
```

### 5.3 文档管理（Document Management）

配合 docstore 可以追踪"哪些文档已入库"，实现：
- **`refresh_ref_docs()`**：只重新摄取**内容有变化**的文档（按 hash 去重）
- **删除**：`index.delete_ref_doc("file_id", delete_from_docstore=True)`

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

documents = SimpleDirectoryReader("./data", filename_as_id=True).load_data()
index = VectorStoreIndex.from_documents(documents)

# 第二次调用：内容没变的文档会被跳过
index.refresh_ref_docs(documents)
```

---

## 6. 第三阶段：存储（Storing）

> 官网路径：`Understanding → Storing`、`Module Guides → Storing`

### 6.1 三类存储 + 默认行为

| 组件 | 存什么 | 默认实现 |
|---|---|---|
| **Document Store** | 原始 Document/Node（`SimpleDocumentStore`） | 内存 + JSON 落盘 |
| **Index Store** | 索引结构元信息（`SimpleIndexStore`） | 内存 + JSON 落盘 |
| **Vector Store** | embedding 向量（`SimpleVectorStore`） | 内存 + JSON 落盘 |

> **知识点**：不指定任何存储时，LlamaIndex 把所有东西放在**内存**里，`index.storage_context.persist()` 时写进 `./storage` 目录。这就是"默认存储"。

### 6.2 持久化与恢复（两行保存、两行恢复）

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage
from llama_index.core import StorageContext

# ① 构建并持久化
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.storage_context.persist(persist_dir="./storage")

# ② 下次启动：从磁盘恢复，避免重新 embedding（省钱省时间）
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)
```

### 6.3 使用向量数据库（以 Chroma 为例，官网示例）

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
```

官网列出 30+ 种向量库集成（Chroma、Qdrant、Pinecone、Weaviate、Milvus、PGVector、Elasticsearch……），模式都是同款三步：

```python
vector_store = XxxVectorStore(...)                # ① 初始化向量库
storage_context = StorageContext.from_defaults(vector_store=vector_store)  # ② 装进 StorageContext
index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)  # ③ 构建索引
```

> **知识点**：若向量库是**全托管**（LlamaCloud、Pinecone 等），可以设置 `store_nodes_override=True` 允许把 Node 也存进向量库，让向量库成为唯一事实来源。

---

## 7. 第四阶段：查询（Querying）

> 官网路径：`Understanding → Querying`

官网定义：查询 = **从索引拿到最相关上下文** → **喂给 LLM 合成答案**。Query Engine 内部由三件套组成：

```
Query Engine = Retriever（检索器） + Node Postprocessor（后处理器） + Response Synthesizer（响应合成器）
```

### 7.1 Retriever（检索器）

```python
from llama_index.core.retrievers import VectorIndexRetriever

# 用法一：从索引一键生成
retriever = index.as_retriever(similarity_top_k=10)

# 用法二：手动实例化
retriever = VectorIndexRetriever(index=index, similarity_top_k=10)

nodes = retriever.retrieve("What did the author do growing up?")
for node in nodes:
    print(node.score, node.text[:80])   # score = 相似度分数
```

> **知识点**：Retriever **不调用 LLM**，只返回 Node 列表。想要纯检索（比如接自己的 RAG 管道）就用它，想直接要答案就用 Query Engine。

### 7.2 Node Postprocessor（节点后处理器）

检索之后、合成之前，对结果做过滤/重排/精调：

```python
from llama_index.core.postprocessor import SimilarityPostprocessor, KeywordNodePostprocessor

postprocessors = [
    # 丢弃相似度低于 0.7 的节点
    SimilarityPostprocessor(similarity_cutoff=0.7),
    # 按 must/exclude 关键词过滤
    KeywordNodePostprocessor(required_keywords=["finance"], exclude_keywords=["hyperion"]),
]
```

官网还有 `SentenceEmbeddingOptimizer`（保留相关句）、`CohereRerank`/`LLMRerank`（重排序）、`PIINodePostprocessor`（脱敏）、`MetadataReplacementPostProcessor`（句子窗口检索用）等。

### 7.3 Response Synthesizer（响应合成器）

决定"怎么把多个 chunk 喂给 LLM"：

| 模式 | 行为 |
|---|---|
| `compact`（默认） | 尽量把 chunk 塞满每次调用，最少化 LLM 调用次数 |
| `refine` | 逐个 chunk 迭代精炼答案（先答一段，再拿下一段修订） |
| `tree_summarize` | 两两合并成树状汇总 |
| `simple_summarize` | 截断所有文本一次性摘要 |
| `no_text` | 只做检索不生成回答 |

```python
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode

synth = ResponseSynthesizer.from_args(
    response_mode=ResponseMode.COMPACT,
    llm=llm,
    streaming=True,      # 流式输出
)
query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
```

### 7.4 Router（路由器）

有多个工具/引擎时，让 LLM 根据问题自动选择：

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        # 一个查小说、一个查传记
        query_engine_tool_for_novels,
        query_engine_tool_for_biographies,
    ],
)
response = router_query_engine.query("哪部小说最出名？")   # 自动路由到小说引擎
```

### 7.5 高级组合：单步查询引擎让 Agent 决定检索路径

官网展示了 `SimpleMultiModalQueryEngine`、`SubQuestionQueryEngine`（把复杂问题拆成子问题分别检索再汇总）与 `QueryPipeline`（声明式管道编排）：

```python
from llama_index.core.query_pipeline import QueryPipeline

qp = QueryPipeline(modules={...}, verbose=True)
qp.add_modules({"retriever": retriever, "synth": synth})
qp.add_link("retriever", "synth")
response = qp.run(query_str="...")
```

---

## 8. 第五阶段：评估与可观测性

> 官网路径：`Module Guides → Evaluating / Observability`

- **可观测性**：一行 `Settings.set_global_handler("wandb")`（或 Arize Phoenix、Langfuse 等回调）即可把每次 LLM 调用、检索结果、Token 消耗全量上报到追踪面板。核心抽象是 `CallbackManager`。
- **评估模块**：内置 Query 评估（`FaithfulnessEvaluator` 忠实度 / `RelevancyEvaluator` 相关性）、Retrieval 评估（MRR、Hit Rate）、合成测试数据集生成器（`RagDatasetGenerator`）。
- 官网评估理念：**真实答案 vs 生成答案**、**检索质量**两条线都要评。

---

## 9. Agent（智能体）

> 官网路径：`Understanding → Building an agent`

**定义**：Agent 是由 LLM 驱动的**半自治软件**：接收任务 → 推理规划 → 反复"思考 + 调用工具" → 直到完成任务。与 Query Engine 的区别：Query Engine 是固定流程（检索→生成），Agent 能**自己决定下一步做什么**。

### 9.1 两种内置 Agent（官网 Understanding/Agent 专页）

| Agent | 循环机制 | 说明 |
|---|---|---|
| **FunctionAgent**（新，推荐） | 基于函数调用（Function Calling/Tool Use） | 要求 LLM 支持 tool call（OpenAI、Anthropic、Gemini 等） |
| **ReActAgent**（经典） | "Reason + Act"：先写 `Thought/Action/Action Input/Observation` 文本再解析 | 对不支持函数调用的模型也能用 |

```python
from llama_index.core.agent import ReActAgent, FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def multiply(a: float, b: float) -> float:
    """Useful for multiplying two numbers."""
    return a * b

tool = FunctionTool.from_defaults(fn=multiply)

# 新版 FunctionAgent
agent = FunctionAgent(
    tools=[tool],
    llm=OpenAI(model="gpt-4o-mini"),
)
response = await agent.run("What is 1234 * 4567?")
print(response)

# 经典 ReActAgent（同步接口）
agent = ReActAgent.from_tools([tool], llm=OpenAI(model="gpt-4o-mini"))
print(agent.chat("What is 1234 * 4567?"))
```

### 9.2 多轮对话与记忆

```python
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    tools=[tool],
    memory=ChatMemoryBuffer.from_defaults(token_limit=4096),  # 会话记忆
    verbose=True,
)
agent.chat("我叫小明")
agent.chat("我叫什么名字？")   # 能记住上文
```

### 9.3 流式输出

```python
agent = FunctionAgent(tools=[tool], llm=OpenAI("gpt-4o-mini"))
handler = agent.run("Write a haiku about work")
async for ev in handler.stream_events():     # AgentWorkflow 事件流
    if hasattr(ev, "delta"):
        print(ev.delta, end="", flush=True)  # 逐 token 输出
response = await handler
```

### 9.4 Agent 的记忆与上下文升级

官网 Agent 页列出三种上下文机制：
1. **SimpleChatMemory / ChatMemoryBuffer** —— 滚动保存最近对话；
2. **VectorMemory** —— 对话历史向量化，按相关性召回（长对话必备）；
3. **`Memory`（新版统一抽象）** —— `Memory.from_defaults()`，支持 token 截断 + 数据库后端（SQLite/Redis）。

### 9.5 完整例子：给 Agent 配一个 RAG 工具

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 先建一个查询引擎
index = VectorStoreIndex.from_documents(SimpleDirectoryReader("data").load_data())
query_engine = index.as_query_engine(similarity_top_k=5)

# 把查询引擎包装成 Agent 工具
tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="about_author",
        description="回答关于作者成长经历的所有问题",
    ),
)

agent = ReActAgent.from_tools([tool], llm=llm, verbose=True)
print(agent.chat("作者小时候做过哪些工作？"))
```

---

## 10. Tools（工具）

> 官网路径：`Module Guides → Agent → Tools`

**概念**：Tool 是 Agent 可调用的函数/能力的抽象。LlamaIndex 把工具分为 5 类（官网四类 + 工具规范）：

| 类别 | 说明 | 典型例子 |
|---|---|---|
| **FunctionTool** | 任意 Python 函数包装成工具 | 上面 multiply 的例子 |
| **QueryEngineTool** | 把查询引擎暴露给 Agent | RAG 工具 |
| **ToolSpec**（工具规范包） | 一次引入一组现成集成工具 | `GmailToolSpec`、`OpenWeatherMapToolSpec` |
| **Utility Tools** | 平台级工具 | `QueryPlanTool`、`OnDemandLoaderTool`（按需读文档）、`LoadAndSearchTool` |
| **Retriever/Reader 工具** | 检索/加载类包装 | `RetrieverTool` |

### 10.1 FunctionTool 三种写法（官网示例）

```python
# 写法一：从任意函数自动生成（自动推断签名+docstring）
from llama_index.core.tools import FunctionTool

def calculate_circle_area(radius: float) -> float:
    """计算圆的面积。"""
    return 3.14159 * radius ** 2

tool = FunctionTool.from_defaults(fn=calculate_circle_area)
# LLM 看到的工具描述 = 函数名 + docstring + 类型注解

# 写法二：显式指定 schema（更可控）
from llama_index.core.tools import FunctionTool
from pydantic import BaseModel, Field

class WeatherSchema(BaseModel):
    city: str = Field(description="城市名，如：北京")
    unit: str = Field(default="celsius", description="温度单位")

def get_weather(city: str, unit: str = "celsius") -> str:
    """获取指定城市的当前天气。"""
    return f"{city} 晴，25{unit[0].upper()}"

weather_tool = FunctionTool.from_defaults(
    fn=get_weather,
    name="get_weather",
    description="查询城市实时天气",
    fn_schema=WeatherSchema,
)

# 写法三：异步函数（agent.run 是异步的）
async def async_tool_fn(x: int) -> int:
    """异步工具示例"""
    return x * 2
```

> **知识点**：`return_direct=True` 可以让工具结果**直接作为最终答案返回**（跳过 LLM 再加工），适合确定性输出（如计算器、数据库查询）。

### 10.2 ToolSpec：一行接入第三方服务

```python
from llama_index.tools.gmail import GmailToolSpec

tool_spec = GmailToolSpec()
# tool_spec 含一组工具：搜索邮件、读邮件、写草稿、发邮件……
tools = tool_spec.to_tool_list()

from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
agent.chat("帮我看看最近的未读邮件都说了什么")
```

官网收录的 ToolSpec：Gmail、Google Calendar、GitHub、Slack、Notion、OpenWeatherMap、SQL（NL2SQL）、Tavily/Brave/Bing 搜索、Yelp、Zapier 等。

### 10.3 工具与 Agent 的组合模式（官网 Understanding/Agent）

- **`QueryEngineTool`** + 多个索引 = Agent 自己决定查哪个知识库（比 Router 更灵活）；
- **`OnDemandLoaderTool`**：Agent 运行时才去读文档（省 Token）；
- **`QueryPlanTool`**：让 Agent 先制定查询计划再并行执行子查询。

---

## 11. Workflows（事件驱动工作流）

> 官网路径：`llamaagents → Workflows`（独立的 `llama-index-workflows` 包，已内置 `llama-index-core`）

**定义**：Workflow 是**事件驱动、基于步骤（step-based）**的程序流控制方式。每个应用由若干 step 组成，step 之间通过**事件**通信。这是 LlamaIndex 当前主推的多 Agent 编排方式（旧的 `AgentWorkflow` 已迁移到 Workflows 库）。

### 11.1 最小示例

```python
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent, StopEvent

# 自定义事件（携带数据在 step 间流转）
class MyEvent(Event):
    msg: list[str]

class MyWorkflow(Workflow):
    @step
    async def step_one(self, ev: StartEvent) -> MyEvent:
        print(ev.query)                      # 启动参数
        return MyEvent(msg=["Step 1 done"])

    @step
    async def step_two(self, ev: MyEvent, ctx: Context) -> StopEvent:
        ctx.store["some_key"] = "value"      # Context 共享状态
        msg = ev.msg + ["Step 2 done"]
        return StopEvent(result=msg)         # 触发 StopEvent 即结束

wf = MyWorkflow(timeout=30, verbose=False)
result = await wf.run(query="初始输入")     # → ["Step 1 done", "Step 2 done"]
```

### 11.2 核心规则

- 一个 Workflow **必须有且只有一个** `StartEvent` 接收 step 和一个返回 `StopEvent` 的 step（`StopEvent` 可带 `result`，即 `wf.run()` 的返回值）；
- step 间传递数据靠**事件**，跨 step 共享状态用 `Context.store`；
- **并发**：多个 step 监听同一事件时并发执行；`@step(num_workers=3)` 可让单 step 并行处理多条事件；
- **流式**：`async for ev in handler.stream_events()` 逐事件获取进度；
- **绘图**：`wf.draw_all_possible_flows("flow.html")` 自动画出流程图；
- **人机协同**：事件可等待外部输入（human-in-the-loop）；
- 可作为 **Agent 的一种实现方式**（官网：agent = workflow + tool calling + memory）。

---

## 12. LLM / Embedding 配置（Settings）

> 官网路径：`Understanding → Using LLMs`、`Module Guides → Models`

### 12.1 全局配置 vs 局部配置

```python
# 全局：影响之后所有模块
from llama_index.core import Settings
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

# 局部：只影响单个组件
index = VectorStoreIndex.from_documents(docs, embed_model=local_embed)  # 只改索引 embedding
qe = index.as_query_engine(llm=Settings.llm)                            # 只改查询 LLM
```

### 12.2 Embedding 用法（官网 Using LLMs 页附带的 Embedding 示例）

```python
from llama_index.embeddings.openai import OpenAIEmbedding

embed_model = OpenAIEmbedding(model="text-embedding-3-small")
embeddings = embed_model.get_text_embedding("Hello World!")   # → [0.1, -0.2, ...] 1536 维
```

### 12.3 调试利器：MockLLM 拦截真实调用

```python
from llama_index.core.llms import MockLLM
from llama_index.core import Settings

# 让所有 LLM 调用返回固定内容——调试 RAG 流程时不用烧 token
Settings.llm = MockLLM(default_response="这是模拟回答")

qe = index.as_query_engine()
print(qe.query("随便问点什么"))
```

> **知识点**：官网 Tips 特别推荐：**调试检索流程时用 MockLLM**，可以零成本看到检索到的 chunk 是什么；等确认检索没问题再换回真 LLM。

---

## 13. 结构化输出（Structured Output）

> 官网路径：`Module Guides → Query Engine → Structured Outputs`

用 Pydantic 模型约束 LLM 输出，供下游程序消费：

```python
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List

# ① 定义输出模型
class Biography(BaseModel):
    """作者小传。"""
    name: str = Field(description="作者姓名")
    key_events: List[str] = Field(description="人生关键事件列表")
    summary: str = Field(description="一句话总结")

# ② 查询引擎直接指定 response_model
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(
    response_model=Biography,
    output_cls=Biography,
)
bio = query_engine.query("根据文档总结作者信息")
print(type(bio))          # <class 'Biography'> —— 已是结构化对象
print(bio.name, bio.key_events)

# ③ 或者用 StructuredOutputQueryEngine 手动组装
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_engine import StructuredOutputQueryEngine

output_parser = PydanticOutputParser(Biography)
qe = index.as_query_engine()
structured_qe = StructuredOutputQueryEngine.from_args(
    query_engine=qe,
    output_parser=output_parser,
    output_cls=Biography,
)
bio = structured_qe.query("...")
```

官网还提供 **PydanticProgram** 系列（`LLMTextCompletionProgram`、OpenAI JSON mode、`OpenAIPydanticProgram`）做"Prompt → Pydantic 对象"转换，以及 `StructuralOutputQueryEngine` 配合**合成器级**结构化输出。

---

## 14. 进阶主题速览

官网 `Module Guides` 其余章节一览（知道有什么、需要时再查）：

| 主题 | 要点 |
|---|---|
| **Prompts** | 所有内置提示词模板都可覆盖：`qe.get_prompts()` 查看 → `update_prompts()` 替换；`PromptTemplate` 支持 `{var}` 占位与富格式 |
| **Multi-Modal（多模态）** | `MultiModalLLM`（GPT-4V 等）+ 图文混合索引，`SimpleDirectoryReader` 直接读图片 |
| **Chat Engine** | `index.as_chat_engine(chat_mode="condense_question"/"react"/"best")` 一行获得多轮对话 + RAG |
| **Query Pipeline** | 声明式 DAG 编排（组件 + 连线），替代手写胶水代码，支持 JSON 定义 |
| **Node Parser 进阶** | `SentenceWindowNodeParser`（句子窗口）、`HierarchicalNodeParser`（层级，配 AutoMergingRetriever）、`DocumentSummaryIndex` |
| **Retriever 进阶** | `AutoMergingRetriever`、`RecursiveRetriever`（引用块→父文档）、`FusionRetriever`（混合检索+RRF） |
| **Deploying** | `llama-deploy` 部署为服务；`AgentRunner`/`AgentWorker` 抽象自定义 Agent 循环 |
| **Observability** | 一行接入 Arize Phoenix / Langfuse / W&B / OpenTelemetry 全链路追踪 |
| **Evaluation** | `FaithfulnessEvaluator`（回答是否忠于上下文）、`RelevancyEvaluator`、检索 Hit-Rate/MRR、`RagDatasetGenerator` 自动造评测集 |
| **LlamaCloud** | `LlamaParse`（复杂 PDF 解析）、`LlamaCloudIndex`（托管 RAG）、`LlamaCloudRetriever` |
| **TypeScript** | `npm install @llama-index/core`，API 与 Python 大体对齐 |

---

## 15. 推荐学习路径

官网文档结构（左侧导航三大板块）：

```
Getting Started（入门）        → 安装、启动教程、高级概念、Using LLMs
Understanding（理解·必读）      → 按五阶段组织的系列短教程：使用 LLM → Agent → RAG（加载/索引/存储/查询）
Module Guides（模块指南·参考）  → 每个模块的完整 API 细节
```

**建议顺序**：

1. 跑通 5 行代码 RAG（§3.3）；
2. 通读 High-Level Concepts 建立对象模型（§3.2）；
3. 按 Understanding 的 RAG 五阶段各写一个小例子（§4-§7）；
4. 学习 Agent + Tools + Workflows（§9-§11）；
5. 需要生产化时再查 Module Guides：IngestionPipeline 缓存/并行（§4.6）、向量库接入（§6.3）、评估与可观测性（§8、§14）。

**常用官方链接**：

- 文档主页：<https://developers.llamaindex.ai/python/framework>
- 快速入门教程（OpenAI 版）：<https://developers.llamaindex.ai/python/framework/getting_started/starter_example>
- 高级概念：<https://developers.llamaindex.ai/python/framework/getting_started/concepts>
- 索引原理图解：<https://developers.llamaindex.ai/python/framework/module_guides/indexing/index_guide>
- Agent 工具列表：<https://developers.llamaindex.ai/python/framework/module_guides/deploying/agents/tools>
- Workflows：<https://developers.llamaindex.ai/python/llamaagents/workflows>
- LlamaHub（数据连接器/工具市场）：<https://llamahub.ai>
- GitHub：<https://github.com/run-llama/llama_index>
