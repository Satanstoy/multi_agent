
# AI 法律咨询助手 (多智能体 CrewAI + RAG) ⚖️

本项目是一个基于 CrewAI 框架构建的多智能体 AI 法律咨询助手。它利用大型语言模型 (LLM) 和检索增强生成 (RAG) 技术，模拟法律专业人士处理用户咨询的流程。

## ✨ 主要特性

* **多智能体架构:** 采用三级 Agent 设计（协调员、工具执行员、回复生成员），明确各阶段任务，实现更精细化的处理流程。
* **检索增强生成 (RAG):** 集成了基于 `Langchain` 和 `ChromaDB` 的 RAG 功能，可以通过 `法律条款检索工具(LAS)` 从本地知识库中检索相关法律条文。
* **可定制工具:** 定义了多种法律相关工具（如类案匹配、罪名预测、法条检索、网络搜索等，部分为占位符），易于扩展和实现具体功能。
* **可配置 LLM 后端:** 通过 `config.py` 和环境变量配置连接到任何兼容 OpenAI API 的 LLM 服务（例如本地部署的 vLLM）。通过 `LiteLLM` 进行实际调用，模型名称可能需要指定提供商前缀 (如 `openai/your-model-path`) 以确保正确路由。
* **结构化工作流:** 使用 CrewAI 的 `Sequential` 流程定义任务执行顺序，确保逻辑清晰、可追溯。
* **本地知识库支持:** 包含创建本地法律文档向量索引的脚本，支持 `.docx` 文件。

## 📂 项目结构




本项目是一个基于 CrewAI 框架构建的多智能体 AI 法律咨询助手。它利用大型语言模型 (LLM) 和检索增强生成 (RAG) 技术，通过结构化的多 Agent 协作，模拟法律专业人士处理用户咨询的流程。

## ✨ 主要特性

* **多智能体架构:** 采用三级 Agent 设计（协调员、工具执行员、回复生成员），明确各阶段任务，实现更精细化的处理流程。
* **检索增强生成 (RAG):** 集成了基于 `Langchain` 和 `ChromaDB` 的 RAG 功能，可以通过 `法律条款检索工具(LAS)` 从本地知识库中检索相关法律条文。
* **可定制工具:** 定义了多种法律相关工具（如类案匹配、罪名预测、法条检索、网络搜索等，部分为占位符），易于扩展和实现具体功能。
* **可配置 LLM 后端:** 通过 `config.py` 和环境变量配置连接到任何兼容 OpenAI API 的 LLM 服务（例如本地部署的 vLLM）。通过 `LiteLLM` 进行实际调用，模型名称可能需要指定提供商前缀 (如 `openai/your-model-path`) 以确保正确路由。
* **结构化工作流:** 使用 CrewAI 的 `Sequential` 流程定义任务执行顺序，确保逻辑清晰、可追溯。
* **本地知识库支持:** 包含创建本地法律文档向量索引的脚本，支持 `.docx` 文件。

## 📂 项目结构
```
multi_agent/
├── agents/                 # 定义 Agent 角色、目标和背景故事
│   └── legal_agents.py     # 包含法律咨询协调员、工具执行员和回复生成员 Agent
├── legal_docs/             # 存放用于 RAG 的法律文档知识库
│   ├── index_legal_docs.py # 用于处理 legal_docs 中的 .docx 文档并创建向量索引的脚本
│   └── .docx法律文件        # 示例：放入法律文件，如 "合同法案例.docx"
├── tools/                  # 定义 Agent 可以使用的工具
│   └── legal_tools.py      # 包含 RAG 法条检索工具和其他占位符工具
├── workflow/               # 定义任务 (Tasks) 和工作流 (Crew)
│   └── legal_workflow.py   # 创建包含 Agents 和 Tasks 的 Crew
├── legal_db/               # (自动生成) RAG 向量数据库的存储目录
├── models/                 # 存放本地模型文件 (例如嵌入模型)
│   └── m3e-base/           # (示例) 嵌入模型 m3e-base 的文件存放处
├── config.py               # 配置 LLM 连接参数和实例化 LLM 客户端
├── main.py                 # 项目入口，运行交互式命令行界面
└── README.md               # 本文件
```

## ⚙️ 先决条件

在开始之前，请确保您已准备好以下环境和资源：

1.  **Python 环境:** Python 3.8 或更高版本。
2.  **LLM 服务端点:** 一个正在运行且兼容 OpenAI API 的大型语言模型服务端点。
    * 例如，您可以使用 vLLM 在本地部署一个开源模型 (如 Qwen, Llama 等)。
    * 确保您有该端点的 URL (例如 `http://localhost:8000/v1`) 和 API 密钥 (如果需要)。
3.  **嵌入模型文件:**
    * 本项目默认配置使用本地嵌入模型 (如 `m3e-base`) 进行 RAG。
    * 请预先下载模型文件，并将其放置在项目根目录下的 `models/m3e-base/` 目录中 (如果使用其他模型或路径，请相应修改 `legal_docs/index_legal_docs.py` 和 `tools/legal_tools.py` 中的路径配置)。
4.  **Git (可选):** 用于克隆本仓库。

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

本项目通过 config.py 连接 LLM，主要通过环境变量进行配置。请在您的操作系统中设置以下环境变量：

LLM_MODEL: 您希望使用的模型名称或路径。
重要提示: LiteLLM 通常需要模型名称带有提供商前缀。例如，如果您使用本地 vLLM 部署的 /data/sj/models/Qwen3-8B 模型，此处应设置为类似 openai//data/sj/models/Qwen3-8B (注意双斜杠表示绝对路径，单斜杠可能是相对路径或 Hugging Face 模型名，具体请参考 LiteLLM 文档和您的 LiteLLM 配置)。
LLM_BASE_URL: LLM 服务的 API 端点 (例如 http://localhost:8000/v1)。
LLM_API_KEY: 访问 LLM 服务的 API 密钥 (如果您的 LLM 服务不需要密钥，可以设置为任意非空字符串，如 "not-needed")。
或者，您可以直接修改 config.py 文件中的默认值，但不推荐用于生产环境。

### 5. 准备 RAG 知识库

* 确保项目根目录下已创建 legal_docs 文件夹。
* 将您的法律文档（.docx 格式）放入 legal_docs 文件夹中。

### 6. 创建 RAG 索引

```bash
python legal_docs/index_legal_docs.py
```

### 7. 运行项目

```bash
python main.py
```

### 📖 使用示例 (Usage Example)
启动 main.py 后，您将看到欢迎信息和输入提示：

```
============================================================
⚖️  欢迎使用 AI 法律咨询助手 (模拟版) ⚖️
  (输入 '退出' 或 'exit' 来结束程序)
============================================================

👉 请您描述遇到的法律问题：
```

在此提示符后输入您的问题，例如：“我想咨询关于租房合同提前解约的问题。” 然后按回车键。AI 助手将开始处理您的请求，并经过一系列内部 Agent 的协作后给出回复。您可以继续对话，AI 会考虑之前的交流历史。

## ⚙️ 工作原理

#### 1. 用户通过命令行界面 (main.py) 输入法律问题。
#### 2. execute_workflow 函数被调用，该函数根据用户输入和对话历史创建并启动 CrewAI 工作流。
#### 3. 法律咨询协调员 (Agent 1 - legal_coordinator):
   - 接收用户当前问题和对话历史。
   - 分析信息是否充分，目标是否明确。
   - 输出一个标准指令字符串，决定下一步行动（例如：'需要澄清'、'使用工具回答: LAS'、'无需工具直接回答' 或 '生成结束语'）。
#### 4. 法律工具执行专员 (Agent 2 - legal_tool_executor_agent):
   - 接收协调员的指令。
   - 如果指令是调用工具：则根据用户原始提问准备参数并执行指定的工具（例如，调用法条检索工具 legal_article_search_rag）。工具执行后，将原始结果 (Observation) 传递给下一个 Agent。
   - 如果指令是其他类型（如 '需要澄清'、'无需工具直接回答'、'生成结束语'）：则直接将该指令字符串作为其 Final Answer 传递。
#### 5. 法律回复整合与生成专员 (Agent 3 - legal_response_synthesizer_agent):
   - 接收来自工具执行专员的输出（可能是工具的 Observation，也可能是协调员传递下来的指令字符串）。
   - 结合用户原始提问、对话历史以及协调员的意图。
   - 如果输入是工具结果 (Observation)，则将其整合、提炼并组织成通俗易懂、面向用户的回复。
   - 如果输入是 '需要澄清' 指令，则基于原始问题和历史对话，生成具体的澄清问题给用户。
   - 如果输入是 '无需工具直接回答' 指令，则基于通用法律知识和对话上下文生成直接答案。
   - 如果输入是 '生成结束语' 指令，则生成礼貌的结束语。
   - 最终生成纯净的文本回复。
#### 6. main.py 对最终回复进行一些可能的清理后，展示给用户。


## 🔧 配置项

* config.py: 核心配置文件。用于设置 LLM 的 API 地址、模型名称和 API 密钥（主要通过环境变量读取）。必须正确配置才能运行项目。
* legal_docs/index_legal_docs.py: RAG 知识库的索引脚本。配置包括源文档目录 (SOURCE_DIRECTORY)、嵌入模型路径 (EMBEDDING_MODEL_LOCAL_PATH)、向量数据库持久化路径 (PERSIST_DIRECTORY)。运行此脚本以构建或更新本地知识库。
* tools/legal_tools.py: 定义了 Agent 可用的工具。RAG 工具 (legal_article_search_rag) 的配置（如向量数据库路径、嵌入模型路径）也在此文件中指定，并应与索引脚本的配置保持一致。
* agents/legal_agents.py: 定义了各个 Agent 的角色 (Role)、目标 (Goal)、背景故事 (Backstory) 和所使用的 LLM 实例。这是调整 Prompt 和 Agent 核心行为逻辑的关键文件。
* workflow/legal_workflow.py: 定义了每个 Agent 执行的任务 (Tasks) 以及这些任务如何串联成一个完整的工作流 (Crew)。

## 💡 TODO / 未来改进

- [ ] 实现占位工具: 为 similar_case_matching, legal_charge_prediction, legal_element_recognition, legal_event_detection, legal_text_summary 等工具接入真实的模型或 API 服务。
- [ ] 优化 Prompt工程: 持续迭代优化各 Agent 的 Prompt，以提高准确性和回复质量。
- [ ] 改进回复清理逻辑: 完善 main.py 中对 LLM 输出的后处理和清理规则。
- [ ] 测试: 增加单元测试和集成测试，确保代码质量和稳定性。