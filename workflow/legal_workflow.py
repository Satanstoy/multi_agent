# workflow/legal_workflow.py
from crewai import Task, Crew, Process
from agents.legal_agents import legal_coordinator, legal_executor
from tools.legal_tools import available_tools # 导入工具列表以获取工具名称

def create_legal_crew(user_input: str, conversation_history: str = "无历史对话") -> Crew:
    """
    创建并配置用于处理单个法律咨询请求的 Crew。
    :param user_input: 用户当前的法律问题输入。
    :param conversation_history: (可选) 此前的对话历史记录。
    :return: 配置好的 Crew 实例。
    """
    # 动态获取工具名称列表，可以用于提示协调员 Agent
    tool_names = ", ".join([tool.name for tool in available_tools])

    # --- 定义任务 (Tasks) ---

    # 任务1: 由协调员执行，分析输入并做出决策
    decision_task = Task(
        description=f"""
        请仔细分析以下用户提问和对话历史记录。
        ---
        用户当前提问: "{user_input}"
        ---
        对话历史: "{conversation_history}"
        ---
        你可以考虑使用的工具有: {tool_names}
        ---
        你的任务是：根据你的角色（法律咨询协调员）和目标，判断处理该用户提问的最佳下一步策略。
        请输出以下三种明确的决策指令之一：
        1. '需要澄清' (如果信息不足或模糊)
        2. '无需工具直接回答' (如果信息充分且无需工具)
        3. '使用工具回答: [工具名称1], [工具名称2]' (如果需要使用工具，请列出工具名称，最多两个)
        请确保你的输出仅仅是这三种指令中的一个字符串。
        """,
        agent=legal_coordinator,
        expected_output="一个决策指令字符串，例如：'需要澄清' 或 '无需工具直接回答' 或 '使用工具回答: 相似案例匹配工具, 法律条款检索工具'"
    )

    # 任务2: 由执行者执行，根据协调员的决策行动
    execution_task = Task(
        description=f"""
        现在，你需要根据协调员给出的决策指令来执行下一步操作。
        ---
        协调员的决策指令是: {{context}}  (这里会自动填充上一个任务 decision_task 的输出)
        ---
        原始用户提问是: "{user_input}"
        ---
        对话历史是: "{conversation_history}"
        ---
        你的任务是：
        - 如果指令是 '需要澄清'，请生成一个友好且具体的追问，向用户询问所需信息。
        - 如果指令是 '无需工具直接回答'，请基于你的法律知识生成一份初步的法律建议或信息回复。
        - 如果指令是 '使用工具回答: ...'，请：
            1. 识别出需要使用的工具名称。
            2. 使用你拥有的同名工具 ({tool_names}) 进行调用，输入可以是原始用户提问或相关案情。
            3. 整合工具返回的结果和你的知识，生成最终的法律建议或解答。
        ---
        **重要**: 你的最终输出应该是直接发送给用户的完整回复文本。不要包含任何内部思考过程或指令文本。
        """,
        agent=legal_executor,
        context=[decision_task], # 本任务的输入依赖于上一个任务的输出
        expected_output="最终面向用户的文本回复（可能是提问，也可能是法律建议）。"
    )

    # --- 创建工作组 (Crew) ---
    legal_crew = Crew(
        agents=[legal_coordinator, legal_executor], # 按顺序定义参与的 Agent
        tasks=[decision_task, execution_task],     # 按顺序定义要执行的任务
        process=Process.sequential,                # 定义任务执行流程为顺序执行
        verbose=True, # 确保这里是 True
        # memory=True # 如果需要跨多轮对话保持记忆，可以启用 memory
    )

    print("-" * 30)
    print("工作流 (Crew) 创建完成。")
    print(f"  -包含 Agents: {[agent.role for agent in legal_crew.agents]}")
    print(f"  -包含 Tasks: {[task.description[:50]+'...' for task in legal_crew.tasks]}")
    print(f"  -流程类型: {legal_crew.process}")
    print("-" * 30)

    return legal_crew