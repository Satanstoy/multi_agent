# main.py
import json
import traceback
from workflow.legal_workflow import create_legal_crew # 从工作流模块导入 Crew 创建函数

# --- 工作流执行封装 ---
def execute_workflow(user_input: str) -> str:
    """
    为给定的用户输入初始化并运行法律咨询工作流。
    :param user_input: 用户输入的法律问题。
    :return: 工作流执行后最终生成的面向用户的回复文本，或错误信息。
    """
    try:
        # 对于简单的单轮交互，历史可以就是当前问题
        # 如果需要多轮，这里需要维护一个更复杂的对话历史状态
        conversation_history = f"用户刚刚问了：{user_input}"

        # 为本次请求创建 Crew 实例
        # 注意：每次用户输入都创建一个新的 Crew 实例，除非你启用了 memory 并希望状态持续
        workflow_crew = create_legal_crew(user_input, conversation_history)

        print("\n🚀 开始执行工作流 (Kicking off the workflow)...")

        # 启动工作流
        # inputs 字典用于传递需要在 Task 描述中动态填充的变量
        # 在我们的 create_legal_crew 函数中，user_input 和 history 已直接嵌入 Task 描述
        # 如果 Task description 使用了 {variable_name} 占位符，则需要在这里提供
        # result = workflow_crew.kickoff(inputs={'variable_name': value})
        result = workflow_crew.kickoff() # 在本例中，不需要额外 inputs

        print("✅ 工作流执行完毕 (Workflow finished).")

        # 检查 result 是否为 None 或非字符串
        if isinstance(result, str):
             return result.strip()
        elif result is None:
             print("⚠️ 工作流返回了 None，可能意味着没有产生预期输出。")
             return "抱歉，处理过程中没有得到明确的回复。请尝试重新提问或调整问题描述。"
        else:
             print(f"⚠️ 工作流返回了非字符串类型：{type(result)}")
             # 尝试将其转换为字符串，或提供通用错误消息
             try:
                  return str(result)
             except:
                  return "抱歉，处理过程中遇到未知格式的返回结果。"


    except Exception as e:
        print(f"❌ 工作流执行时发生严重错误: {str(e)}")
        print("详细错误追踪信息:")
        traceback.print_exc()
        # 向用户返回一个友好的错误提示
        return f"抱歉，处理您的请求时遇到了内部错误。请稍后再试。错误信息：{str(e)}"

# --- 主交互循环 ---
def main():
    """
    程序主入口，运行用户交互循环。
    """
    print("=" * 60)
    print("⚖️  欢迎使用 AI 法律咨询助手 (模拟版) ⚖️")
    print("   (输入 '退出' 或 'exit' 来结束程序)")
    print("=" * 60)

    while True:
        try:
            user_input = input("\n👉 请您描述遇到的法律问题：")

            if user_input.strip().lower() in ['退出', 'exit']:
                print("\n👋 感谢您的使用，再见！")
                break

            if not user_input.strip():
                print("⚠️ 输入不能为空，请输入您的问题。")
                continue

            print("\n⏳ 正在为您分析问题，请稍候...")
            print("-" * 60)

            # 调用工作流执行函数
            final_response = execute_workflow(user_input)

            print("-" * 60)
            print("💡 AI 助手的回复:")
            print(final_response)
            print("=" * 60)

        except KeyboardInterrupt:
            print("\n👋 检测到中断操作，程序退出。")
            break
        except Exception as e:
            print(f"\n❌ 主循环发生未预期错误: {e}")
            traceback.print_exc()
            # 可以在这里添加一些错误恢复逻辑或直接退出

if __name__ == "__main__":
    main()