# config.py (配置Y)

import os
import traceback
# 从 langchain_openai 更改为从 langchain_ollama 导入 ChatOllama
from langchain_ollama import ChatOllama # <--- 修改点 1

# --- 设置环境变量 ---
# 确保这些环境变量被设置，供 LiteLLM 使用
# 根据您的第一个代码片段，这里使用 Ollama 的默认端口 11434
LLM_BASE_URL_FOR_ENV = os.getenv("LLM_BASE_URL", "http://localhost:8000") # <--- 修改点 2: 更改为 11434
LLM_API_KEY_FOR_ENV = os.getenv("LLM_API_KEY", "ollama") # Ollama通常不需要API Key，但LiteLLM可能需要非空字符串，这里可以用'ollama'或其他任意字符串

print(f"--- INFO: 正在设置 OPENAI_API_BASE 环境变量为: '{LLM_BASE_URL_FOR_ENV}' ---")
os.environ["OPENAI_API_BASE"] = LLM_BASE_URL_FOR_ENV

print(f"--- INFO: 正在设置 OPENAI_API_KEY 环境变量为: '{LLM_API_KEY_FOR_ENV}' ---")
os.environ["OPENAI_API_KEY"] = LLM_API_KEY_FOR_ENV

# --- LLM 配置 ---
# !!! 关键：模型名称必须带有 'ollama/' 前缀给 LiteLLM !!!
LLM_MODEL_FOR_LITELLM_PROVIDER_ID = os.getenv("LLM_MODEL", "ollama/qwen3:8b") # <--- 修改点 3: 确保是 'ollama/你的模型名称'

# ChatOllama 实例的参数
LLM_BASE_URL_FOR_CHATOLLAMA = LLM_BASE_URL_FOR_ENV # <--- 修改点 4: 对应 ChatOllama 的参数
LLM_API_KEY_FOR_CHATOLLAMA = LLM_API_KEY_FOR_ENV # <--- 修改点 5: 对应 ChatOllama 的参数

# 打印调试信息
# 这部分调试信息主要是针对vLLM的路径，对于Ollama可以直接看LLM_MODEL_FOR_LITELLM_PROVIDER_ID
_actual_model_path_for_ollama = ( # <--- 修改点 6: 变量名更清晰
    LLM_MODEL_FOR_LITELLM_PROVIDER_ID.split('ollama/', 1)[-1]
    if LLM_MODEL_FOR_LITELLM_PROVIDER_ID.startswith("ollama/")
    else LLM_MODEL_FOR_LITELLM_PROVIDER_ID
)

print(f"--- DEBUG: LLM_MODEL environment variable = {os.getenv('LLM_MODEL')} ( defaulting to: {LLM_MODEL_FOR_LITELLM_PROVIDER_ID} for LiteLLM provider ID) ---")
print(f"--- DEBUG: (Actual model name for Ollama: {_actual_model_path_for_ollama}) ---") # <--- 修改点 7
print(f"--- DEBUG: LLM_BASE_URL used for ChatOllama = {LLM_BASE_URL_FOR_CHATOLLAMA} ---") # <--- 修改点 8
print(f"--- DEBUG: LLM_API_KEY used for ChatOllama = '{LLM_API_KEY_FOR_CHATOLLAMA}' ---") # <--- 修改点 9
print(f"--- DEBUG: OPENAI_API_BASE environment variable is set to: '{os.environ.get('OPENAI_API_BASE')}' ---")
print(f"--- DEBUG: OPENAI_API_KEY environment variable is set to: '{os.environ.get('OPENAI_API_KEY')}' ---")

# --- 使用 LangChain 实例化 LLM ---
llm = None

try:
    # !!! 关键：直接使用 ChatOllama !!!
    print(f"--- 正在初始化 LangChain ChatOllama LLM (连接到 {LLM_BASE_URL_FOR_CHATOLLAMA}, 模型 {LLM_MODEL_FOR_LITELLM_PROVIDER_ID}) ---") # <--- 修改点 10
    llm = ChatOllama( # <--- 修改点 11: 改为 ChatOllama
        model=LLM_MODEL_FOR_LITELLM_PROVIDER_ID,  # <-- 使用带前缀的名称
        base_url=LLM_BASE_URL_FOR_CHATOLLAMA, # <--- 修改点 12: ChatOllama 使用 base_url
        # api_key 参数对于 ChatOllama 通常不是必须的，因为Ollama通常不需要API Key
        # 如果需要，请根据LiteLLM文档或Ollama配置添加
        # request_timeout=60 # ChatOllama 默认没有这个参数，如果需要可能要在 LiteLLM 层配置
        temerature=0.1
    )
    print("✅ LangChain ChatOllama LLM 初始化成功!") # <--- 修改点 13
    print(f" 实例模型名称 (llm.model): {getattr(llm, 'model', 'N/A')}") # <--- 修改点 14: ChatOllama 用 .model
    print(f" 实例API Base (llm.base_url): {getattr(llm, 'base_url', 'N/A')}") # <--- 修改点 15: ChatOllama 用 .base_url
except Exception as e:
    print(f"❌ 初始化 LangChain ChatOllama LLM 时出错: {e}") # <--- 修改点 16
    print(f" 错误类型: {type(e).__name__}")
    print(" 请检查:")
    print(f" 1. Ollama API 服务 ({LLM_BASE_URL_FOR_CHATOLLAMA}) 是否正在运行且可访问?") # <--- 修改点 17
    print(f" 2. 环境变量 OPENAI_API_BASE 和 OPENAI_API_KEY 是否已正确设置 (尽管 Ollama 不强制要求 API KEY)?") # <--- 修改点 18
    print(f" 3. 模型名称 '{LLM_MODEL_FOR_LITELLM_PROVIDER_ID}' 是否适合 LiteLLM 识别 provider (即 'ollama/qwen3:8b' 格式)?")
    print("详细错误追踪信息:")
    traceback.print_exc()
    print("⚠️ LLM 实例未能创建，后续 CrewAI 工作流将无法运行。")

# --- 打印最终配置概览 ---
print("-" * 40)
print("LLM 配置加载完成 (使用 LangChain)")

if llm:
    model_used_in_instance = getattr(llm, 'model', 'N/A') # <--- 修改点 19
    base_url_used_in_instance = getattr(llm, 'base_url', 'N/A') # <--- 修改点 20
    api_key_in_env = os.environ.get("OPENAI_API_KEY", "未设置")
    api_base_in_env = os.environ.get("OPENAI_API_BASE", "未设置")
    env_var_status_desc = f"环境变量 OPENAI_API_KEY='{api_key_in_env}', OPENAI_API_BASE='{api_base_in_env}'"

    print(f" LLM实例模型名 (llm.model): {model_used_in_instance}")
    print(f" LLM实例基础URL (llm.base_url): {base_url_used_in_instance}")
    print(f" 环境变量状态: {env_var_status_desc}")
    print(" 状态: 初始化成功")
else:
    model_to_print = LLM_MODEL_FOR_LITELLM_PROVIDER_ID
    print(f" 尝试使用模型 (Model for LiteLLM): {model_to_print} (尝试配置)")
    print(f" 基础 URL (Base URL): {LLM_BASE_URL_FOR_CHATOLLAMA} (尝试配置)") # <--- 修改点 21
    api_key_in_env = os.environ.get("OPENAI_API_KEY", "未设置")
    api_base_in_env = os.environ.get("OPENAI_API_BASE", "未设置")
    env_var_status_desc = f"环境变量 OPENAI_API_KEY='{api_key_in_env}', OPENAI_API_BASE='{api_base_in_env}'"
    print(f" 环境变量状态: {env_var_status_desc}")
    print(" 状态: 初始化失败 ❌")
    print("-" * 40)

# --- 文件结束 ---