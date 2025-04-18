# config.py
import os
from typing import List, Dict, Any
from openai import OpenAI # 确保安装了 openai 库: pip install openai

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
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            print(f"OpenAI client initialized for model '{model}' at '{base_url}'")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.client = None

    def __call__(self, messages: List[Dict], **kwargs) -> str:
        """
        使得类实例可以像函数一样被调用，用于与 LLM 交互。
        :param messages: 发送给 LLM 的消息列表 (符合 OpenAI 格式)。
        :param kwargs: 其他可能的参数 (在此实现中未使用)。
        :return: LLM 返回的文本内容，或者错误信息。
        """
        if not self.client:
            return "Error: OpenAI client not initialized."
            
        try:
            # 简单的消息清理：确保角色有效，移除连续重复角色的消息
            formatted_messages = [msg for msg in messages if msg.get("role") in ["system", "user", "assistant"]]
            
            cleaned_messages = []
            last_role = None
            for msg in formatted_messages:
                # 检查 content 是否为 None 或非字符串，如果是，则跳过或使用默认值
                if msg.get("content") is None or not isinstance(msg.get("content"), str):
                   print(f"Warning: Skipping message with invalid content: {msg}")
                   continue # 跳过无效消息
                   
                if msg["role"] != last_role:
                    cleaned_messages.append(msg)
                    last_role = msg["role"]
                # 可选：如果连续消息角色相同，可以选择合并内容或保留最后一个
                # else: # 如果角色相同，替换掉上一条相同角色的消息 (CrewAI可能自己会处理)
                #    if cleaned_messages:
                #        cleaned_messages[-1] = msg 

            # 如果清理后没有消息，则返回错误
            if not cleaned_messages:
                 print("Error: No valid messages to send after cleaning.")
                 return "Error: No valid messages to send."
                 
            # CrewAI 通常会管理消息顺序，确保最后是 user message (如果需要)
            # 在这里，我们相信 CrewAI 的处理

            response = self.client.chat.completions.create(
                model=self.model,
                messages=cleaned_messages,
                temperature=self.temperature,
            )
            
            content = response.choices[0].message.content
            # CrewAI 通常会处理 "Final Answer:" 的解析，这里直接返回原始 content
            return content.strip() if content else "Error: LLM returned empty content."
            
        except Exception as e:
            print(f"LLM API 调用错误: {str(e)}")
            # 对于调用错误，返回明确的错误信息
            return f"Error during LLM call: {str(e)}"

# --- 实例化 LLM ---
# 优先从环境变量获取配置，提供默认值作为备选
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo") # 你可以换成你的 DeepSeek 模型名
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1") # 你的本地 DeepSeek 服务地址
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY") # 你的 API Key (如果是本地部署，可能为 "EMPTY")

# 创建全局 LLM 实例，供其他模块导入使用
llm = DeepSeekLLM(
    model=LLM_MODEL,
    temperature=0.1, # 可以适当调整温度参数
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