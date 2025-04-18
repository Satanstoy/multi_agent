# config.py
import os
from typing import List, Dict, Any
from openai import OpenAI # 确保安装了 openai 库: pip install openai
from copy import deepcopy
import traceback

# 设置环境变量，禁用模型验证
os.environ["LITELLM_SKIP_MODEL_VALIDATION"] = "TRUE"

# 设置环境变量，完全禁用LiteLLM路由
os.environ["LITELLM_DISABLE_ROUTER"] = "TRUE"

# --- LLM Wrapper Class ---
# 这个类封装了与 DeepSeek 或兼容 OpenAI 的 API 的交互逻辑
class DeepSeekLLM:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0):
        """
        初始化 LLM 客户端。
        :param model: 使用的模型名称。
        :param base_url: API 的基础 URL。
        :param api_key: API 密钥。
        :param temperature: 生成文本的温度参数 (0 表示更确定性)。
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        try:
            # 使用OpenAI客户端连接到本地DeepSeek服务
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            print(f"OpenAI client initialized for model '{model}' at '{base_url}'")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None

    def __call__(self, messages: List[Dict], **kwargs) -> str:
        """简化版本的消息重组"""
        if not self.client:
            return "Error: OpenAI client not initialized."
        
        try:
            print("\n=============== 开始调用LLM ===============")
            print(f"原始消息数量: {len(messages)}")
            for i, msg in enumerate(messages):
                print(f"  原始消息 {i+1}: role={msg.get('role', 'unknown')}, content={msg.get('content', '')[:50]}...")
            
            # 收集所有角色的消息内容
            all_content = ""
            for msg in messages:
                if msg.get("content") and isinstance(msg.get("content"), str):
                    role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                    all_content += f"[{role}]: {content}\n\n"
            
            # 创建单个用户消息
            simplified_messages = [{"role": "user", "content": all_content.strip() + "\n\n[assistant]:"}]
            
            print(f"Simplifying message structure to a single user message:")
            print(f"  Content: {simplified_messages[0]['content'][:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=simplified_messages,
                temperature=self.temperature,
            )
            
            content = response.choices[0].message.content
            print(f"调用结果: {content[:100]}...")
            print("=============== LLM调用完成 ===============\n")
            return content.strip() if content else "Error: LLM returned empty content."
        
        except Exception as e:
            print("\n=============== LLM调用错误 ===============")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误消息: {str(e)}")
            traceback.print_exc()  # 打印完整的堆栈跟踪
            print("=============== 错误详情结束 ===============\n")
            return f"Error during LLM call: {str(e)}"

# --- 实例化 LLM ---
# 使用通用模型名称，不指定具体模型
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo") # 使用一个常见的模型名称
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1") 
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-no-key-required")

# 创建全局 LLM 实例，供其他模块导入使用
llm = DeepSeekLLM(
    model=LLM_MODEL,
    temperature=0.1,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY
)

# 打印初始化信息，确认配置加载情况
print("-" * 30)
print(f"LLM 配置加载:")
print(f"  模型 (Model): {LLM_MODEL}")
print(f"  基础 URL (Base URL): {LLM_BASE_URL}")
print(f"  API Key 使用: {'是' if LLM_API_KEY != 'EMPTY' else '否 (或使用默认占位符)'}")
print("-" * 30)

# 在Agent实例化时，修改执行方法
def custom_execute_task(self, task, context=None, **kwargs):
    """直接调用OpenAI客户端而不是通过LiteLLM，并打印Agent输出"""
    try:
        # 这里直接使用DeepSeekLLM实例而不是Agent的llm属性
        from config import llm

        print(f"\n===== Agent: {self.role} 开始执行任务 =====") # 添加 Agent 角色打印
        print(f"任务描述 (片段): {task.description[:100]}...") # 打印任务描述（可选）
        if context:
            print(f"上下文 (片段): {str(context)[:100]}...") # 打印上下文（可选）

        # 构建任务提示
        task_prompt = task.description
        if context:
            # 确保上下文是字符串，因为 decision_task 的输出是字符串
            task_prompt = task_prompt.format(context=str(context))

        # 创建消息
        messages = [
            {"role": "system", "content": f"你是{self.role}。{self.goal}"},
            {"role": "user", "content": task_prompt}
        ]

        # 直接调用我们的DeepSeekLLM实例
        result = llm(messages) # result 已经是最终的字符串输出了

        print(f"\n💡 Agent: {self.role} 的输出:") # 添加 Agent 输出标记
        print(result)                              # 打印 Agent 的完整输出
        print(f"===== Agent: {self.role} 任务执行完毕 =====") # 添加结束标记

        return result # 返回结果供 crewai 流程继续

    except Exception as e:
        print(f"❌ Agent: {self.role} 执行任务时出错: {e}")
        import traceback
        traceback.print_exc() # 打印详细错误
        return f"Task execution failed for agent {self.role}: {str(e)}"

# 替换Agent的execute_task方法
from crewai.agent import Agent
Agent.execute_task = custom_execute_task