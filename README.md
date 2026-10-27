# learn-llamaindex

基于 [LlamaIndex 官方文档](https://developers.llamaindex.ai/python/framework)（2026-10 版）整理的中文学习笔记。

## 内容

| 文档 | 说明 |
|---|---|
| [llamaindex-知识点整理.md](./llamaindex-知识点整理.md) | 官网使用说明的知识点体系：核心概念、RAG 五阶段（加载/索引/存储/查询/评估）、Agent、Tools、Workflows、Settings 配置、结构化输出等，每个知识点配代码示例 |
| [llamaindex-实战案例集.md](./llamaindex-实战案例集.md) | 10 个可直接运行的实战案例：文档问答、聊天机器人、简历抽取、多文档对比、路由、全能 Agent、报告工作流、句子窗口检索、国产/本地模型、增量摄取 |

## 知识点速览

```
RAG 五阶段：Loading → Indexing → Storing → Querying → Evaluating
核心对象：  Document / Node / Reader / Index / Retriever / QueryEngine / ChatEngine / Agent / Tool
Agent 机制：FunctionAgent（函数调用）· ReActAgent（思维链）· Workflows（事件驱动编排）
```

五行代码版 RAG（官网招牌示例）：

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()   # ① 加载
index = VectorStoreIndex.from_documents(documents)       # ② 索引
query_engine = index.as_query_engine()                   # ③ 查询引擎
response = query_engine.query("What did the author do growing up?")
print(response)                                          # ④⑤ 检索 + 生成
```

## 环境

```bash
pip install llama-index            # 完整包
# 或按需安装
pip install llama-index-core llama-index-llms-openai llama-index-embeddings-openai llama-index-readers-file
```

## 官方资源

- 文档主页：<https://developers.llamaindex.ai/python/framework>
- 官方示例 Notebook：<https://github.com/run-llama/llama_index/tree/main/docs/examples>
- LlamaHub（连接器/工具市场）：<https://llamahub.ai>
- GitHub 仓库：<https://github.com/run-llama/llama_index>

> 小技巧：官方文档任意页面 URL 后加 `index.md` 可获取纯 Markdown 源码。
