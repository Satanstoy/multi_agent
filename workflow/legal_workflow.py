# workflow/legal_workflow.py
from config import llm # 导入llm实例
from crewai import Task, Crew, Process
from agents.legal_agents import (
    legal_coordinator,
    legal_tool_executor_agent,
    legal_response_synthesizer_agent
)

def create_legal_crew(user_input: str, conversation_history: str = "无历史对话") -> Crew:
    """
    创建并配置用于处理单个法律咨询请求的 Crew。
    采用三Agent结构：协调员 -> (条件性)工具执行员 -> 回复整合员。
    :param user_input: 用户当前的法律问题输入。
    :param conversation_history: (可选) 此前的对话历史记录。
    :return: 配置好的 Crew 实例。
    """

    # 任务1: 由协调员执行，分析输入并做出决策
    decision_task = Task(
        description=f"""
        作为法律咨询协调员，请严格遵循你在 Agent Goal 中定义的行动决策强制规则和最终输出格式要求。
        分析以下用户提问和对话历史：
        ---
        用户当前提问: "{user_input}"
        ---
        对话历史: "{conversation_history}"
        ---
        你的任务是：基于你在 Agent Goal 中被设定的详细判断标准，精确判断处理该用户提问的最佳下一步策略，并输出相应的标准指令字符串。
        【特别注意】：你的整个回复**只能是**你在 Agent Goal 中被告知的那四种标准指令字符串之一。不要包含任何其他文字、解释或思考过程。
        """,
        agent=legal_coordinator,
        expected_output="一个标准指令字符串，例如：`'使用工具回答: LAS'` 或 `'需要澄清'` 或 `'无需工具直接回答'` 或 `'生成结束语'`。"
    )

    # 任务2: 由工具执行专员处理，根据协调员的决策决定是否调用工具
    # 它会接收 decision_task 的输出作为 {context}
    conditional_tool_execution_task = Task(
        description=f"""
        现在，你（法律工具执行专员）需要严格根据协调员给出的决策指令（包含在 {{context}} 中）来行动。
        用户的原始提问是: "{user_input}"
        对话历史是: "{conversation_history}"

        你的任务是：
        - 如果协调员指令是 `'使用工具回答: TOOL_ABBR'`，则解析并执行相应的工具，输出工具调用所需的 Thought/Action/Action Input 格式。
        - 如果协调员指令是 `'需要澄清'`、`'无需工具直接回答'` 或 `'生成结束语'`，则直接将该指令作为你的 `Final Answer` 输出，以便传递给下一个Agent。
        严格遵循你在 Agent Goal 中关于这两种情况的输出格式要求。
        """,
        agent=legal_tool_executor_agent,
        context=[decision_task], # 接收来自协调员决策任务的上下文
        expected_output="如果调用工具，则是工具调用格式；否则是协调员的原始指令字符串（如 `'需要澄清'`）作为Final Answer。"
    )

    # 任务3: 由回复整合与生成专员生成最终回复
    # 它会接收 conditional_tool_execution_task 的输出作为 {context}
    # 注意：如果上一步是工具调用，CrewAI 会自动将 Observation 传递过来，
    # 如果上一步是 Final Answer (传递指令)，则那个 Final Answer 的内容就是这里的 {context}。
    # legal_response_synthesizer_agent 的 prompt 需要能够处理这两种输入。
    final_response_generation_task = Task(
        description=f"""
        现在，你（法律回复整合与生成专员）需要根据上一个Agent（工具执行专员）的输出结果（包含在 {{context}} 中），以及最初协调员的指令（也隐含在{{context}}中，如果它是被传递下来的指令的话）、用户的原始提问和对话历史，来生成最终的、直接面向用户的回复。
        用户的原始提问是: "{user_input}"
        对话历史是: "{conversation_history}"
        协调员最初的指令意图需要你从 {{context}} 中判断：
        - 如果 {{context}} 是 `'需要澄清'`、`'无需工具直接回答'` 或 `'生成结束语'`，则按这些指令生成回复。
        - 如果 {{context}} 是工具执行后的 `Observation` (通常是一个字典或结构化文本)，则你需要结合原始用户问题和协调员的工具使用意图（例如，如果调用了LAS工具，说明协调员想查找法条），来整合 `Observation` 并生成回复。

        你的任务是严格按照你在 Agent Goal 中被设定的指令处理规则（特别是关于判断输入是“非工具指令”还是“工具执行结果Observation”并据此生成不同类型回复的逻辑）来执行。
        确保你的最终输出给用户的文本是纯净的，不包含任何内部处理标签。
        """,
        agent=legal_response_synthesizer_agent,
        context=[conditional_tool_execution_task], # 接收来自工具执行任务的上下文
        expected_output="最终的、直接面向用户的纯净文本回复。"
    )

    # --- 创建工作组 (Crew) ---
    legal_crew = Crew(
        agents=[
            legal_coordinator,
            legal_tool_executor_agent,
            legal_response_synthesizer_agent
        ],
        tasks=[
            decision_task,
            conditional_tool_execution_task,
            final_response_generation_task
        ],
        llm=llm,
        process=Process.sequential,
        verbose=True,
        # memory=True # 按需启用
    )

    print("-" * 30)
    print("工作流 (Crew) 创建完成。")
    print(f"  -包含 Agents: {[agent.role for agent in legal_crew.agents]}")
    print(f"  -包含 Tasks: {[task.description.split('---')[0].strip() for task in legal_crew.tasks]}") # 调整打印，避免过长
    print(f"  -流程类型: {legal_crew.process}")
    print("-" * 30)

    return legal_crew