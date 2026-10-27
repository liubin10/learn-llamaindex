# LlamaIndex 实战案例集

> 配套文档：[llamaindex-知识点整理.md](./llamaindex-知识点整理.md)
> 案例基于官方 Use Cases 板块（Q&A / Chatbots / Structured Extraction / Agents）+ Understanding 的"Putting it all together"章节整理，代码为 Python，可直接运行（按案例注释安装依赖）。
> 通用前置：`pip install llama-index llama-index-readers-file`，并设置 `os.environ["OPENAI_API_KEY"] = "sk-..."`（案例九演示如何换成国产/本地模型，免 OpenAI）。

---

## 目录

| # | 案例 | 使用的核心能力 |
|---|---|---|
| 一 | 个人文档知识库问答（RAG 入门） | SimpleDirectoryReader、VectorStoreIndex、QueryEngine |
| 二 | 企业知识库聊天机器人（多轮对话） | ChatEngine、context 模式、记忆 |
| 三 | 简历信息结构化抽取 | as_structured_llm、PydanticProgram |
| 四 | 多文档对比问答（子问题分解） | SubQuestionQueryEngine |
| 五 | 多知识库路由 | RouterQueryEngine |
| 六 | 全能 Agent（计算器+检索+联网） | FunctionAgent、FunctionTool、QueryEngineTool |
| 七 | 事件驱动多步骤工作流（自动写报告） | Workflows（@step + 事件） |
| 八 | 句子窗口检索（提升 RAG 精度） | SentenceWindowNodeParser、MetadataReplacement |
| 九 | 国产/本地模型全流程 | Ollama、HuggingFaceEmbedding、OpenAILike |
| 十 | 生产级增量摄取管道 | IngestionPipeline、缓存、UPSERTS |

---

## 案例一：个人文档知识库问答（RAG 入门）

**场景**：把一堆 PDF/Word/Markdown 丢进 `data/` 目录，对它们提问，获得带上下文的回答。这是官网 Q&A 用例的标准形态。

```python
# pip install llama-index
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

os.environ["OPENAI_API_KEY"] = "sk-..."

PERSIST_DIR = "./storage"

def build_or_load_index():
    # 二次运行时直接从磁盘加载索引，省去重新 embedding 的钱和时间
    if os.path.exists(PERSIST_DIR):
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        return load_index_from_storage(storage_context)

    documents = SimpleDirectoryReader("data").load_data()   # 支持 pdf/docx/md/txt...
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)  # 持久化
    return index

index = build_or_load_index()

# 一次性问答
query_engine = index.as_query_engine(similarity_top_k=5)
response = query_engine.query("这份文档的核心结论是什么？")
print(response)

# 查看引用来源（检索到的原文 chunk）
for node in response.source_nodes:
    print(f"[score={node.score:.3f}] {node.metadata.get('file_name')}: {node.text[:60]}...")
```

**关键点**：
- `similarity_top_k` 控制召回多少个 chunk，默认 2，问答建议 3~5；
- `response.source_nodes` 拿到命中的原文和分数，可用于前端"显示引用"；
- 官网 Q&A 用例页强调：数据无需训练，RAG 只在查询时动态注入上下文。

**变体——语义搜索（不要 LLM 回答，只要最相关段落）**：

```python
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve("查询词")
for n in nodes:
    print(n.score, n.text[:100])
```

---

## 案例二：企业知识库聊天机器人（多轮对话）

**场景**：客服/内部助手，记住上下文，能追问"你刚才说的第一点展开讲讲"。对应官网 Chatbots 用例 + Chat Engine 模块。

```python
# pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# chat_mode="context"：每轮对话先检索知识库再回答
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=4096),
    system_prompt=(
        "你是公司的产品手册助手，只根据知识库内容回答，"
        "回答完可以引导用户继续提问。"
    ),
    verbose=True,
)

# 多轮对话循环
while True:
    user_input = input("你: ")
    if user_input in ("退出", "exit", "q"):
        break
    response = chat_engine.chat(user_input)
    print(f"助手: {response}\n")
```

**三种 chat_mode 选型**（官网 Chat Engine 文档）：

| 模式 | 行为 | 适合 |
|---|---|---|
| `condense_question` | 先把对话历史+新问题合成一个独立问题再检索 | 简单多轮 |
| `context`（推荐） | 每轮都检索知识库，检索结果与历史一起生成 | 知识库问答机器人 |
| `react` | 用 ReAct Agent 驱动，可调用多种工具 | 需要工具的对话 |

**命令行聊天工具（一行搭建）**：

```python
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
index.as_chat_engine(chat_mode="context").start_chat()  # 官方 CLI 聊天界面
```

---

## 案例三：简历信息结构化抽取

**场景**：把非结构化的简历/合同/发票文本抽成 JSON，直接入库。对应官网 Structured Data Extraction 用例，是 LlamaIndex 的招牌能力之一。

```python
# pip install llama-index
from llama_index.core.llms import ChatMessage
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional

llm = OpenAI(model="gpt-4o-mini")

# ① 定义抽取目标（Pydantic 模型即输出 Schema）
class WorkExperience(BaseModel):
    company: str = Field(description="公司名")
    title: str = Field(description="职位")
    years: int = Field(description="工作年限")

class Resume(BaseModel):
    name: str = Field(description="姓名")
    phone: Optional[str] = Field(None, description="电话")
    skills: List[str] = Field(description="技能列表")
    work_experience: List[WorkExperience] = Field(description="工作经历")

# ② 一行把普通 LLM 变成结构化 LLM
structured_llm = llm.as_structured_llm(Resume)

# ③ 喂入原始简历文本
resume_text = open("resume.txt", encoding="utf-8").read()
resp = structured_llm.chat([
    ChatMessage(role="user", content=f"抽取以下简历中的信息：\n{resume_text}")
])
print(type(resp.raw))     # <class 'Resume'> —— Pydantic 对象
print(resp.raw.name)
print(resp.raw.skills)
print(resp.raw.model_dump_json(indent=2))   # 直接序列化为 JSON
```

**变体——从整个文档库批量抽取**（官网 extraction 用例的完整形态）：

```python
from llama_index.core import Document
from llama_index.core.program import LLMTextCompletionProgram

# Prompt 模板 + Pydantic 输出 → 抽取程序
program = LLMTextCompletionProgram.from_defaults(
    output_cls=Resume,
    prompt_template_str=(
        "{resume_text}\n"
        "从上面的简历中抽取结构化信息。"
    ),
    verbose=True,
)
result = program(resume_text=resume_text)
```

**变体——RAG + 结构化输出结合**（先检索再抽取，适合超长文档）：

```python
class DocSummary(BaseModel):
    title: str
    key_points: List[str]

query_engine = index.as_query_engine(response_model=DocSummary)
summary: DocSummary = query_engine.query("总结这份文档")
print(summary.title, summary.key_points)
```

---

## 案例四：多文档对比问答（子问题分解）

**场景**：问"Uber 的营收增速比 Lyft 高多少？"——答案分散在两份文档里，单索引检索必然丢信息。`SubQuestionQueryEngine` 会自动把问题拆成子问题、分别检索、再汇总。

```python
# pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# 两份文档各建一个索引
docs_uber = SimpleDirectoryReader(input_files=["data/uber_10k.pdf"]).load_data()
docs_lyft = SimpleDirectoryReader(input_files=["data/lyft_10k.pdf"]).load_data()

uber_qe = VectorStoreIndex.from_documents(docs_uber).as_query_engine()
lyft_qe = VectorStoreIndex.from_documents(docs_lyft).as_query_engine()

# 包装成工具，description 越清楚，子问题路由越准
tools = [
    QueryEngineTool(
        query_engine=uber_qe,
        metadata=ToolMetadata(
            name="uber_10k",
            description="Uber 2021 年财报，包含营收、司机数等数据",
        ),
    ),
    QueryEngineTool(
        query_engine=lyft_qe,
        metadata=ToolMetadata(
            name="lyft_10k",
            description="Lyft 2021 年财报，包含营收、司机数等数据",
        ),
    ),
]

s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=tools)
response = s_engine.query("Uber 和 Lyft 的营收增速分别是多少？谁增长更快？")
print(response)
```

**运行时会发生什么**（官网示例的真实输出结构）：

```
Generated 3 sub questions.
[uber_10k] Q: Uber 2021 年营收是多少？        → A: $17.46B (83% yoy growth)
[lyft_10k] Q: Lyft 2021 年营收是多少？        → A: $3.2B (36% yoy growth)
[compare]  汇总两个答案生成最终对比结论
```

**适用**：跨文档对比、复合问题（"A 的价格和 B 的保修政策"）。代价是 LLM 调用次数 = 子问题数 + 1。

---

## 案例五：多知识库路由

**场景**：系统里同时有"小说文档库"和"传记文档库"（或"产品手册"和"财务报表"），让 LLM 根据问题自动选对知识库。

```python
# pip install llama-index
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata

novels_index = VectorStoreIndex.from_documents(
    SimpleDirectoryReader("data/novels").load_data())
bio_index = VectorStoreIndex.from_documents(
    SimpleDirectoryReader("data/biographies").load_data())

novels_tool = QueryEngineTool(
    query_engine=novels_index.as_query_engine(),
    metadata=ToolMetadata(
        name="novels",
        description="回答与小说内容相关的问题",
    ),
)
bio_tool = QueryEngineTool(
    query_engine=bio_index.as_query_engine(),
    metadata=ToolMetadata(
        name="biographies",
        description="回答与作者生平传记相关的问题",
    ),
)

router = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[novels_tool, bio_tool],
)
# 自动路由到 novels 引擎
print(router.query("作者最出名的小说是哪一部？"))
# 自动路由到 biographies 引擎
print(router.query("作者是在哪一年出生的？"))
```

**选型**：`LLMSingleSelector`（选一个引擎）、`LLMMultiSelector`（同时选多个）；把 selector 换成 `PydanticSingleSelector` 走 OpenAI function calling 更稳。

---

## 案例六：全能 Agent（计算器 + 检索 + 自定义工具）

**场景**：官网 Agents 用例的典型组合——一个 Agent 同时会算数、会查知识库、会调天气 API，自己决定用哪个。

```python
# pip install llama-index llama-index-llms-openai
import asyncio
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import FunctionAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI

# 工具 1：普通函数
def multiply(a: float, b: float) -> float:
    """两个数相乘。"""
    return a * b

def add(a: float, b: float) -> float:
    """两个数相加。"""
    return a + b

# 工具 2：RAG 查询引擎包装成工具
docs = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(docs)
rag_tool = QueryEngineTool(
    query_engine=index.as_query_engine(),
    metadata=ToolMetadata(
        name="author_docs",
        description="回答与作者生平、作品相关的问题",
    ),
)

agent = FunctionAgent(
    tools=[
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        rag_tool,
    ],
    llm=OpenAI(model="gpt-4o-mini"),
)

async def main():
    resp = await agent.run("1234 乘以 4567 等于多少？再帮我查一下作者的出生年份")
    print(resp)
    # Agent 会先调用 multiply 工具算数，再调用 author_docs 检索，最后汇总

asyncio.run(main())
```

**关键点**：
- 工具的 `description`/docstring 是 Agent 的决策依据，必须写清楚；
- `return_direct=True` 的工具（如计算器）结果直接返回，不再让 LLM 复述；
- 需要更多现成工具时用 ToolSpec，例如 `pip install llama-index-tools-gmail` 后 `GmailToolSpec().to_tool_list()` 直接获得读写邮件等十余个工具。

---

## 案例七：事件驱动多步骤工作流（自动写报告）

**场景**：官方 Workflows 文档的经典示例——"检索资料 → 生成报告大纲 → 逐节写内容"的多步骤流水线。Workflows 是 LlamaIndex 目前主推的编排方式。

```python
# pip install llama-index   （workflows 已内置）
import asyncio
from workflows import Workflow, step, Context
from workflows.events import Event, StartEvent, StopEvent

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.openai import OpenAI

# 步骤之间传递的事件
class OutlineEvent(Event):
    outline: str

class SectionEvent(Event):
    section_content: str

class ReportWorkflow(Workflow):
    @step
    async def retrieve(self, ev: StartEvent, ctx: Context) -> OutlineEvent:
        # 步骤 1：检索资料，生成大纲
        index = VectorStoreIndex.from_documents(
            SimpleDirectoryReader("data").load_data()
        )
        context = index.as_query_engine().query(
            ev.topic + "：请列出写报告需要的背景要点"
        )
        llm = OpenAI(model="gpt-4o-mini")
        outline = llm.complete(
            f"为主题「{ev.topic}」写一个三节报告的大纲，参考要点：{context}"
        ).text
        ctx.store["topic"] = ev.topic          # 跨步骤共享状态
        return OutlineEvent(outline=outline)

    @step
    async def write_section(self, ev: OutlineEvent, ctx: Context) -> SectionEvent:
        # 步骤 2：按大纲写正文
        llm = OpenAI(model="gpt-4o-mini")
        content = llm.complete(
            f"主题「{ctx.store.get('topic')}」，大纲：{ev.outline}，请写出正文。"
        ).text
        return SectionEvent(section_content=content)

    @step
    async def finalize(self, ev: SectionEvent) -> StopEvent:
        # 步骤 3：收尾输出
        return StopEvent(result={"report": ev.section_content})

wf = ReportWorkflow(timeout=120, verbose=True)
result = asyncio.run(wf.run(topic="LLM 检索增强生成的实践"))
print(result["report"])
```

**关键点**：
- step 之间靠**事件**驱动：返回什么事件，监听该事件的 step 就被触发；
- `ctx.store` 是跨 step 的共享黑板；
- 多个 step 监听同一事件会**并发执行**（可并行写三个章节）；
- 调试神器：`wf.draw_all_possible_flows("flow.html")` 直接生成流程图；
- 加 `@step(num_workers=3)` 可让一个 step 并行消费多条事件。

---

## 案例八：句子窗口检索（提升 RAG 精度的进阶技巧）

**场景**：直接用 512 字的大 chunk 做 embedding 会稀释语义。句子窗口方案：**按句子做 embedding，命中后自动把前后句展开**返回给 LLM——检索更准、上下文更完整。

```python
# pip install llama-index
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings,
    StorageContext, load_index_from_storage,
)
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor  # 备选导入路径

# ① 构造"句子窗口"解析器：每个句子一个 node，附带前后 3 句的窗口
node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

documents = SimpleDirectoryReader("data").load_data()
nodes = node_parser.get_nodes_from_documents(documents)
sentence_index = VectorStoreIndex(nodes)

# ② 查询时：用 MetadataReplacementPostProcessor 把句子替换成窗口全文
query_engine = sentence_index.as_query_engine(
    similarity_top_k=5,
    node_postprocessors=[
        MetadataReplacementPostProcessor(target_metadata_key="window"),
        # 可再加一层重排，进一步提精度
        SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-6-v2", top_n=3),
    ],
)
print(query_engine.query("作者对编程的态度是什么？"))
```

**关键点**：
- embedding 用单句（语义聚焦），生成用窗口（上下文完整），两头兼得；
- 同族技巧还有 `HierarchicalNodeParser` + `AutoMergingRetriever`（大块检索小块验证，自动向上合并父节点），适合长文档。

---

## 案例九：国产/本地模型全流程（免 OpenAI）

**场景**：内网部署或省成本。用 Ollama 跑 LLM，用 BGE 中文 embedding，全程本地。

```bash
# ① 本地准备
ollama pull llama3.1        # 或 qwen2.5、glm4 等
pip install llama-index-llms-ollama llama-index-embeddings-huggingface
```

```python
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.llm = Ollama(model="llama3.1", request_timeout=120.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-zh-v1.5")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
print(index.as_query_engine().query("文档讲了什么？"))
```

**变体——调用 DeepSeek / Qwen / GLM 等 OpenAI 兼容 API**：

```python
# pip install llama-index-llms-openai-like
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

Settings.llm = OpenAILike(
    model="deepseek-chat",
    api_base="https://api.deepseek.com/v1",
    api_key="sk-...",
    is_chat_model=True,
)
```

**关键点**：`Settings` 全局配置一次，所有 Reader/Index/QueryEngine/Agent 默认使用；单个组件想用别的模型，构造时传 `llm=` 或 `embed_model=` 局部覆盖即可。

---

## 案例十：生产级增量摄取管道

**场景**：文档库每天更新，不想每次全量重建索引。`IngestionPipeline` + docstore 缓存实现"只处理新增/变更的文档"。

```python
# pip install llama-index
from llama_index.core.ingestion import (
    IngestionPipeline,
    DocstoreStrategy,
    IngestionCache,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.cache import SimpleCache
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.embeddings.openai import OpenAIEmbedding

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=20),
        OpenAIEmbedding(),
    ],
    docstore=SimpleDocumentStore(),
    docstore_strategy=DocstoreStrategy.UPSERTS,   # 新增/更新/删除都同步
    cache=IngestionCache(cache=SimpleCache()),    # 变换结果缓存，重跑零成本
)

documents = SimpleDirectoryReader("data", filename_as_id=True).load_data()
pipeline.update_docstore(documents)               # 登记/更新 docstore
nodes = pipeline.run(documents=documents, show_progress=True, num_workers=4)

# 基于管道产物建索引并持久化 docstore
from llama_index.core import StorageContext
index = VectorStoreIndex(nodes)
index.storage_context.persist("./storage")
pipeline.docstore.persist("./docstore.json")

# 下次有新文档：只把新文件放进 documents 再跑一遍 run()，旧文档自动跳过
```

**关键点**：
- `filename_as_id=True` 让文档按文件名去重；
- 想感知"哪些文档真的变了"，用 `index.refresh_ref_docs(documents)` 返回每个文档是否更新；
- `num_workers>1` 开多进程并行做切分和 embedding，大批量导入时显著提速。

---

## 官方更多实战资源

| 资源 | 链接 |
|---|---|
| Use Cases 总览 | <https://developers.llamaindex.ai/python/framework/use_cases> |
| Q&A（RAG）用例 | <https://developers.llamaindex.ai/python/framework/use_cases/q_and_a> |
| 聊天机器人用例 | <https://developers.llamaindex.ai/python/framework/use_cases/chatbots> |
| 结构化抽取用例 | <https://developers.llamaindex.ai/python/framework/use_cases/extraction> |
| Agent 用例 | <https://developers.llamaindex.ai/python/framework/use_cases/agents> |
| Q&A 模式合集（语义搜索/摘要/子问题） | <https://developers.llamaindex.ai/python/framework/understanding/putting_it_all_together/q_and_a> |
| 聊天机器人教程 | <https://developers.llamaindex.ai/python/framework/understanding/putting_it_all_together/chatbots/building_a_chatbot> |
| 官方 Notebook 示例仓库 | <https://github.com/run-llama/llama_index/tree/main/docs/examples> |
| LlamaHub（连接器/工具市场） | <https://llamahub.ai> |

> 提示：官方文档任意页面 URL 后加 `index.md` 可获取纯 Markdown 源码，便于对照最新 API。
