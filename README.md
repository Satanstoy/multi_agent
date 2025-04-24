
# AI 法律咨询助手 (多智能体 CrewAI + RAG) ⚖️

本项目是一个基于 CrewAI 框架构建的多智能体 AI 法律咨询助手。它利用大型语言模型 (LLM) 和检索增强生成 (RAG) 技术，模拟法律专业人士处理用户咨询的流程。

## ✨ 主要特性

* **多智能体架构:** 使用两个专门的 Agent (协调员和执行者) 来处理用户请求，实现更结构化的处理流程。
* **检索增强生成 (RAG):** 集成了基于 `Langchain` 和 `ChromaDB` 的 RAG 功能，可以通过 `法律条款检索工具` 从本地知识库中检索相关法律条文。
* **可定制工具:** 定义了多种法律相关工具（部分为占位符），易于扩展和实现具体功能。
* **可配置 LLM 后端:** 通过 `config.py` 文件轻松配置连接到本地（如 DeepSeek）或任何兼容 OpenAI API 的 LLM 服务。
* **结构化工作流:** 使用 CrewAI 的 `Sequential` 流程定义任务执行顺序，确保逻辑清晰。

## 📂 项目结构

```
multi_agent/
├── agents/              # 定义 Agent 角色、目标和背景故事
│   ├── init.py
│   └── legal_agents.py  # 包含法律咨询协调员和执行者 Agent
├── legal_docs/          # 存放用于 RAG 的法律文档知识库
│   └── index_legal_docs.py # 用于处理 legal_docs 中的文档并创建向量索引的脚本
├── tools/               # 定义 Agent 可以使用的工具
│   ├── init.py
│   └── legal_tools.py   # 包含 RAG 法律条款检索工具和其他占位符工具
├── workflow/            # 定义任务 (Tasks) 和工作流 (Crew)
│   ├── init.py
│   └── legal_workflow.py # 创建包含 Agents 和 Tasks 的 Crew
├── legal_db/            # (自动生成) RAG 向量数据库的存储目录
├── config.py            # 配置 LLM 连接参数和实例化 LLM 客户端
├── main.py              # 项目入口，运行交互式命令行界面
└── README.md            # 本文件
```

## 🚀 快速开始

### 1. 环境准备

* 确保你已安装 Python 3.8 或更高版本。
* (可选但推荐) 创建并激活一个虚拟环境：
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 2. 克隆仓库

```bash
git clone <your-repository-url>
cd multi_agent
```

### 3. 安装依赖

根据项目中的 import 语句，创建一个 requirements.txt 文件，至少包含以下核心库（版本可能需要根据你的环境调整）：

```
# requirements.txt
crewai>=0.28.8,<0.29.0
langchain-community
sentence-transformers
chromadb
openai
# pypdf
# python-dotenv
```

然后运行安装命令：

```bash
pip install -r requirements.txt
```

### 4. 配置 LLM 后端

本项目通过 config.py 连接 LLM。你需要一个兼容 OpenAI API 的 LLM 服务端点。

**方式一 (推荐): 设置环境变量**

在你的操作系统中设置以下环境变量：

- `LLM_MODEL`: 模型名称（如 deepseek-chat）。
- `LLM_BASE_URL`: API 端点（如 http://localhost:8000/v1）。
- `LLM_API_KEY`: API 密钥。

**方式二: 直接修改 config.py**

```python
# config.py
LLM_MODEL = os.getenv("LLM_MODEL", "your-model-name")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "your_llm_api_endpoint/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "your_api_key_if_needed")
```

### 5. 准备 RAG 知识库

* 创建 legal_docs 文件夹 (如果不存在)。
* 放入你的法律文档 (.txt 文件)。

### 6. 创建 RAG 索引

```bash
python legal_docs/index_legal_docs.py
```

### 7. 运行项目

```bash
python main.py
```

## ⚙️ 工作原理

1. 用户输入问题。
2. `execute_workflow` 创建 Crew 实例。
3. Agent 1（协调员）分析输入并输出策略。
4. Agent 2（执行者）根据策略：
   - 追问
   - 直接回答
   - 使用工具生成回复
5. 输出结果展示给用户。

## 🔧 配置项

* LLM: config.py
* RAG: legal_docs/index_legal_docs.py, tools/legal_tools.py
* Agent: agents/legal_agents.py
* 工具: tools/legal_tools.py

## 💡 TODO / 未来改进

- [ ] 实现占位工具
- [ ] 集成真实搜索 API
- [ ] 增强记忆功能
- [ ] 优化 Prompt
- [ ] 改进清理逻辑
- [ ] 增强日志处理
- [ ] 更复杂 Agent 协作