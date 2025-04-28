# agents/legal_agents.py
from crewai import Agent
from config import llm  # 共享的 LLM 配置
from tools.legal_tools import (
    similar_case_matching,
    legal_article_search_rag,
    legal_charge_prediction,
    legal_element_recognition,
    legal_event_detection,
    legal_text_summary,
    web_search
)

# --- 可用工具列表 ---
available_tools = [
    similar_case_matching,      # 类案匹配工具 (SCM)
    legal_article_search_rag,   # 法条检索工具 (LAS)
    legal_charge_prediction,    # 罪名预测工具 (LCP)
    legal_element_recognition,  # 法律要素识别工具 (LER)
    legal_event_detection,      # 法律事件检测工具 (LED)
    legal_text_summary,         # 法律摘要工具 (LTS)
    web_search                  # 互联网搜索工具 (WEB)
]

# --- 定义 “智法” 风格的双 Agent 系统 ---

# 1. 法律咨询协调员 Agent
legal_coordinator = Agent(
    role="法律咨询协调员 (Legal Consultation Coordinator)",
    goal="""分析用户提问与历史，判断信息是否充足，并决策下一步行动：
    1. 若缺少关键事实（如人物、行为、时间、地点），则指令 '需要澄清'。
    2. 若基本事实齐备，但存在定义、后果、步骤等不确定，应：
       - 优先使用最多两个最相关的工具，并指令 '使用工具回答: [工具1, 工具2]'。
    3. 若信息充足且能直接回答，则指令 '无需工具直接回答'。
    **最终输出只能是以下三种之一：'需要澄清'、'无需工具直接回答'、'使用工具回答: [工具名, ...]'。
    不允许添加其他解释。""",
    backstory="""你是经验丰富的 AI 法律助理，擅长高效评估提问内容。
    你的任务是精准、快速地推进问题解决，合理安排澄清与工具调用，避免无意义追问。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    max_iter=3
)

# 2. 法律执行与整合者 Agent
legal_executor = Agent(
    role="法律执行与整合者 (Legal Execution and Integrator)",
    goal="""根据协调员的决策执行任务，并面向用户生成正式回复：
    1. 收到 '需要澄清' 指令时，生成专业、明确的澄清提问。
    2. 收到 '无需工具直接回答' 指令时，利用自身知识直接生成清晰、准确的法律建议。
    3. 收到 '使用工具回答: [工具1, 工具2]' 指令时：
       - 调用对应工具，整合工具输出（包含工具标记内容），并生成增强的法律回复。
       - 回复中可引用工具结果，但不得暴露内部处理过程。
    **你的回复必须直接面向用户，专业、自然、清晰，不包含任何关于指令、工具调用过程的说明。""",
    backstory="""你是专业的法律事务执行 AI，具备强大的法律知识和整合能力。
    你严格服从协调员的指令，高效执行工具调用与答案生成，确保回复内容准确且易于理解。""",
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=available_tools,  # 授权调用7个完整工具
    max_iter=5
)

# --- 系统加载提示 ---
print("-" * 30)
print("Agent 模块加载完成。")
print(f"  - 已定义 Agent: {legal_coordinator.role}")
print(f"  - 已定义 Agent: {legal_executor.role}")
print(f"  - 执行者 Agent 配备 {len(available_tools)} 个工具:")
for tool in available_tools:
    print(f"    - {tool.name}")
print("-" * 30)
