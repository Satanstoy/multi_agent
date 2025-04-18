# main.py
import os
# 设置环境变量，禁用模型验证
os.environ["LITELLM_SKIP_MODEL_VALIDATION"] = "TRUE"
# 设置环境变量，完全禁用LiteLLM路由
os.environ["LITELLM_DISABLE_ROUTER"] = "TRUE"

import json
import traceback
import re # 导入正则表达式模块
from workflow.legal_workflow import create_legal_crew # 从工作流模块导入 Crew 创建函数
from crewai.crews.crew_output import CrewOutput # 可能需要导入这个
from crewai.tasks.task_output import TaskOutput # 导入 TaskOutput

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

        # 检查 result 类型并提取最终输出
        final_answer = ""
        raw_output = None # 用于存储原始输出

        if isinstance(result, CrewOutput) and result.tasks_output:
            print(f"ℹ️ 工作流返回 CrewOutput 对象。尝试提取最后一个任务的输出。")
            last_task_output = result.tasks_output[-1]
            if isinstance(last_task_output, TaskOutput):
                # 优先尝试 .raw 属性，通常包含最原始的 LLM 输出字符串
                if last_task_output.raw:
                    raw_output = str(last_task_output.raw).strip()
                    print(f"  提取到的 raw_output: {raw_output[:100]}...")
                # 如果 .raw 为空，尝试 .result 属性
                elif last_task_output.result:
                    raw_output = str(last_task_output.result).strip()
                    print(f"  提取到的 result: {raw_output[:100]}...")
                # 如果都没有，记录原始对象以供调试
                else:
                    print(f"⚠️ 最后一个 TaskOutput 对象中未找到 'raw' 或 'result' 内容: {last_task_output}")
                    raw_output = str(last_task_output) # 尝试将整个对象转为字符串
            else:
                 print(f"⚠️ 最后一个任务输出类型不是 TaskOutput: {type(last_task_output)}")
                 raw_output = str(last_task_output) # 尝试将未知类型转为字符串
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

        # 如果成功提取到任何形式的输出
        if raw_output:
            final_answer = raw_output
        else: # 如果所有尝试都失败
            print("⚠️ 未能从工作流结果中提取有效输出。")
            final_answer = "抱歉，处理过程中未能获取明确的回复。"

        # 确保 final_answer 是字符串
        if not isinstance(final_answer, str):
            final_answer = str(final_answer)

        # --- 改进的清理逻辑 ---
        cleaned_answer = final_answer # 处理前先备份
        original_cleaned_answer_before_any_cleaning = cleaned_answer # 记录最原始的输出

        # 步骤 1: 移除主要的内部指令和引导语前缀
        prefixes_to_remove = [
            # 将更长、更具体的前缀放在前面
            "根据协调员的决策指令，我需要进行澄清。",
            "根据协调员的决策指令\"需要澄清\"，我将进一步询问用户以获取更多信息。",
            "协调员，您需要我澄清什么具体信息吗？请提供更多细节，以便我能够更好地帮助您。",
            "根据协调员的决策指令，需要进行澄清。",
            "根据协调员的决策指令，我需要",
            "根据协调员的决策指令 ",
            # *** 新增常见的澄清引导语 ***
            "为了更好地理解用户的问题，我需要进一步了解以下信息：",
            "为了更好地理解您的问题，我需要进一步了解以下信息：",
            "为了更好地帮助您，我需要了解以下信息：",
            "为了更好地帮助您，请您提供以下信息：",
            # 移除残留的动作描述
            "进行澄清。",
        ]
        prefix_removed = False
        for prefix in prefixes_to_remove:
            # 使用 startswith 进行检查
            if cleaned_answer.strip().startswith(prefix):
                # 计算需要移除的长度
                prefix_len = len(prefix)
                # 从原始 cleaned_answer（包含可能的原始前导空格）中移除
                cleaned_answer = cleaned_answer[cleaned_answer.find(prefix) + prefix_len:].strip()
                # 移除可能残留的逗号、冒号和空格
                cleaned_answer = cleaned_answer.lstrip('，').lstrip(':：').lstrip()
                print(f"  移除了前缀 '{prefix[:30]}...'")
                prefix_removed = True
                break # 只移除第一个匹配的前缀

        # 步骤 2: (可选) 如果移除了前缀，尝试移除可能残留的用户问题复述 (逻辑保持不变)
        if prefix_removed:
             pattern = r"^\s*(用户提问是|您的问题是)\s*[:：]\s*[""']?.*?['""']?\s*(\n|$)"
             match = re.match(pattern, cleaned_answer, re.MULTILINE)
             if match:
                 print(f"  移除了复述的用户问题: '{match.group(0).strip()[:50]}...'")
                 cleaned_answer = cleaned_answer[match.end():].strip()

        # 步骤 3: 如果清理后结果为空，提供通用消息 (逻辑保持不变)
        if not cleaned_answer.strip():
             if any(kw in original_cleaned_answer_before_any_cleaning for kw in ["为了更好", "请问", "能否提供", "需要了解", "请您提供", "什么信息", "哪些细节"]): # 扩展关键词列表
                 print("  ⚠️ 清理后结果为空，但原始输出似乎包含有效问题，尝试回退并仅移除首要前缀。")
                 cleaned_answer = original_cleaned_answer_before_any_cleaning
                 # 尝试移除最可能的前缀组合
                 possible_prefixes = [
                     "根据协调员的决策指令，我需要进行澄清。",
                     "为了更好地理解用户的问题，我需要进一步了解以下信息："
                     # 添加其他可能的组合
                 ]
                 restored = False
                 for pp in possible_prefixes:
                     if cleaned_answer.startswith(pp):
                         cleaned_answer = cleaned_answer[len(pp):].strip().lstrip(':：').lstrip()
                         if cleaned_answer.strip(): # 确保移除后还有内容
                            restored = True
                            break
                 if not restored or not cleaned_answer.strip(): # 如果恢复失败或恢复后仍为空
                    cleaned_answer = "为了更好地帮助您，请您提供更多关于情况的细节。"

             else:
                 cleaned_answer = "为了更好地帮助您，请您提供更多关于情况的细节。"


        final_answer = cleaned_answer

        return final_answer

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