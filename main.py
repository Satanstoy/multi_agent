# main.py
import os
os.environ['LITELLM_LOG'] = 'DEBUG'

import traceback
import re 
from workflow.legal_workflow import create_legal_crew
from crewai.crews.crew_output import CrewOutput
from crewai.tasks.task_output import TaskOutput
import litellm

# --- 设置环境变量 (保持不变) ---
os.environ["LITELLM_SKIP_MODEL_VALIDATION"] = "TRUE"
# os.environ["LITELLM_DISABLE_ROUTER"] = "TRUE"

# --- 工作流执行封装 (修改以接收和格式化历史) ---
def execute_workflow(user_input: str, history_list: list) -> str:
    """
    为给定的用户输入和对话历史初始化并运行法律咨询工作流。
    :param user_input: 用户最新的输入。
    :param history_list: 包含过去对话内容的列表 (例如 ["User: xxx", "AI: yyy", ...])。
    :return: 工作流执行后最终生成的面向用户的回复文本，或错误信息。
    """
    try:
        # 将列表格式的历史转换为适合 Agent prompt 的字符串格式
        formatted_history = "\n".join(history_list)

        # 创建 Crew 实例，传入当前用户输入和格式化后的历史
        workflow_crew = create_legal_crew(user_input, formatted_history)

        print("\n🚀 开始执行工作流 (Kicking off the workflow)...")
        result = workflow_crew.kickoff()
        print("✅ 工作流执行完毕 (Workflow finished).")

        # --- 结果提取和清理逻辑 (保持你之前的改进) ---
        final_answer = ""
        raw_output = None

        # (结果提取逻辑保持不变)
        if isinstance(result, CrewOutput) and result.tasks_output:
            print(f"ℹ️ 工作流返回 CrewOutput 对象。尝试提取最后一个任务的输出。")
            last_task_output = result.tasks_output[-1]
            if isinstance(last_task_output, TaskOutput):
                if last_task_output.raw:
                    raw_output = str(last_task_output.raw).strip()
                    print(f"  提取到的 raw_output: {raw_output[:100]}...")
                elif last_task_output.result:
                    raw_output = str(last_task_output.result).strip()
                    print(f"  提取到的 result: {raw_output[:100]}...")
                else:
                    print(f"⚠️ 最后一个 TaskOutput 对象中未找到 'raw' 或 'result' 内容: {last_task_output}")
                    raw_output = str(last_task_output)
            else:
                 print(f"⚠️ 最后一个任务输出类型不是 TaskOutput: {type(last_task_output)}")
                 raw_output = str(last_task_output)
        elif isinstance(result, str):
             print(f"ℹ️ 工作流直接返回字符串。")
             raw_output = result.strip()
        elif result is None:
             print("⚠️ 工作流返回了 None。")
        else:
             print(f"⚠️ 工作流返回了未知类型：{type(result)}。尝试转换为字符串。")
             try:
                  raw_output = str(result).strip()
             except Exception as e:
                  print(f"  转换为字符串失败: {e}")

        if raw_output:
            final_answer = raw_output
        else:
            print("⚠️ 未能从工作流结果中提取有效输出。")
            final_answer = "抱歉，处理过程中未能获取明确的回复。"

        if not isinstance(final_answer, str):
            final_answer = str(final_answer)

        # --- 改进的清理逻辑 (移除了正则表达式部分) ---
        cleaned_answer = final_answer
        original_cleaned_answer_before_any_cleaning = cleaned_answer
        prefixes_to_remove = [
            "根据协调员的决策指令，我需要进行澄清。", "根据协调员的决策指令\"需要澄清\"，我将进一步询问用户以获取更多信息。",
            "协调员，您需要我澄清什么具体信息吗？请提供更多细节，以便我能够更好地帮助您。", "根据协调员的决策指令，需要进行澄清。",
            "根据协调员的决策指令，我需要", "根据协调员的决策指令 ",
            "为了更好地理解用户的问题，我需要进一步了解以下信息：", "为了更好地理解您的问题，我需要进一步了解以下信息：",
            "为了更好地帮助您，我需要了解以下信息：", "为了更好地帮助您，请您提供以下信息：",
            "进行澄清。",
        ]
        prefix_removed = False # 这个变量现在可能没什么用了，但保留也无妨
        for prefix in prefixes_to_remove:
            # 前缀移除逻辑保持不变 (这部分不使用正则表达式)
            if cleaned_answer.strip().startswith(prefix):
                prefix_len = len(prefix)
                cleaned_answer = cleaned_answer[cleaned_answer.find(prefix) + prefix_len:].strip()
                cleaned_answer = cleaned_answer.lstrip('，').lstrip(':：').lstrip()
                print(f"  移除了前缀 '{prefix[:30]}...'")
                prefix_removed = True # 标记有前缀被移除
                break

        if not cleaned_answer.strip():
             if any(kw in original_cleaned_answer_before_any_cleaning for kw in ["为了更好", "请问", "能否提供", "需要了解", "请您提供", "什么信息", "哪些细节"]):
                 print("  ⚠️ 清理后结果为空，但原始输出似乎包含有效问题，尝试回退并仅移除首要前缀。")
                 cleaned_answer = original_cleaned_answer_before_any_cleaning
                 possible_prefixes = [
                     "根据协调员的决策指令，我需要进行澄清。", "为了更好地理解用户的问题，我需要进一步了解以下信息："
                 ]
                 restored = False
                 for pp in possible_prefixes:
                     if cleaned_answer.startswith(pp):
                         cleaned_answer = cleaned_answer[len(pp):].strip().lstrip(':：').lstrip()
                         if cleaned_answer.strip():
                            restored = True
                            break
                 if not restored or not cleaned_answer.strip():
                    cleaned_answer = "为了更好地帮助您，请您提供更多关于情况的细节。"
             else:
                 cleaned_answer = "为了更好地帮助您，请您提供更多关于情况的细节。"

        final_answer = cleaned_answer
        # --- 清理逻辑结束 ---

        return final_answer

    except Exception as e:
        print(f"❌ 工作流执行时发生严重错误: {str(e)}")
        print("详细错误追踪信息:")
        traceback.print_exc()
        return f"抱歉，处理您的请求时遇到了内部错误。请稍后再试。错误信息：{str(e)}"

# --- 主交互循环 (保持不变) ---
def main():
    """
    程序主入口，运行用户交互循环。
    """
    print("=" * 60)
    print("⚖️  欢迎使用 AI 法律咨询助手 (模拟版) ⚖️")
    print("   (输入 '退出' 或 'exit' 来结束程序)")
    print("=" * 60)

    conversation_history = [] # 初始化对话历史列表
    is_first_turn = True      # 标记是否是第一轮对话

    while True:
        try:
            if is_first_turn:
                user_input = input("\n👉 请您描述遇到的法律问题：")
                if user_input.strip().lower() in ['退出', 'exit']:
                     print("\n👋 感谢您的使用，再见！")
                     break
                if user_input.strip():
                    is_first_turn = False
                else:
                    print("⚠️ 输入不能为空，请输入您的问题。")
                    continue
            else:
                user_input = input("\n💬 您：")

            if user_input.strip().lower() in ['退出', 'exit']:
                print("\n👋 感谢您的使用，再见！")
                break

            if not user_input.strip():
                print("⚠️ 输入不能为空，请输入您的问题或回复。")
                continue

            conversation_history.append(f"User: {user_input}")

            print("\n⏳ 正在分析并生成回复，请稍候...")
            print("-" * 60)

            final_response = execute_workflow(user_input, conversation_history)

            conversation_history.append(f"AI: {final_response}")

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

if __name__ == "__main__":
    main()