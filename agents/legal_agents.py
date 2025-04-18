# agents/legal_agents.py
from crewai import Agent
from config import llm  # 从配置模块导入共享的 LLM 实例
from tools.legal_tools import available_tools # 从工具模块导入工具列表

# --- 定义 "智法" 风格的 Agent ---

# 1. 法律咨询协调员 Agent
legal_coordinator = Agent(
    role="法律咨询协调员 (Legal Consultation Coordinator)",
    goal="""分析用户法律问题与对话历史，判断信息是否足够。
    1. 信息不足则要求澄清 (决策: '需要澄清')。
    2. 信息足够则判断是否需工具：
       a. 无需工具则直接回答 (决策: '无需工具直接回答')。
       b. 需要工具则选择1-2个最合适的工具 (决策: '使用工具回答: [工具名]')。
    """,
    backstory="""你是一位经验丰富的 AI 法律助理，擅长快速评估用户问题的清晰度与信息完整性，
    明确何时需追问、何时可直接解答、何时需借助外部工具，以指导后续处理。""",
    verbose=True,       # 打印 Agent 的思考过程
    allow_delegation=False, # 该 Agent 不会将任务委托给其他 Agent
    llm=llm,            # 使用共享的 LLM 实例
    max_iter=3          # 限制其内部思考/决策的迭代次数
)

# 2. 法律执行与整合者 Agent
legal_executor = Agent(
    role="法律执行与整合者 (Legal Execution and Integrator)",
    goal="""根据协调员的决策指令执行任务，生成最终用户回复。
    1. 若需澄清，则生成引导用户补充信息的提问。
    2. 若无需工具，则基于内置知识生成初步法律建议。
    3. 若需使用工具，则调用指定工具，整合结果与自身知识，生成增强的法律建议或解答，并可注明信息来源。
    **最终输出必须是直接面向用户的文本回复。**""",
    backstory="""你是一个高效的法律事务执行 AI，精准理解上级指令。你熟练调用法律工具（案例、法条、搜索等），
    并将工具结果与自身知识结合，生成清晰、规范、易懂的法律回复。""",
    verbose=True,
    allow_delegation=False, # 执行者通常也不再向下委托
    llm=llm,
    tools=available_tools, # **关键**: 执行者需要被明确赋予它能使用的工具列表
    max_iter=5          # 可能需要多次迭代来调用工具、思考、整合和生成最终回复
)

print("-" * 30)
print("Agent 模块加载完成。")
print(f"  - 定义了 Agent: {legal_coordinator.role}")
print(f"  - 定义了 Agent: {legal_executor.role}")
print(f"  - 执行者 Agent 被赋予了 {len(legal_executor.tools)} 个工具。")
print("-" * 30)