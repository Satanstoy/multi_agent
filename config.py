# config.py (配置Y)
import os
import traceback
from langchain_openai import ChatOpenAI

# --- 设置环境变量 ---
# 确保这些环境变量被设置，供 LiteLLM 使用
LLM_BASE_URL_FOR_ENV = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY_FOR_ENV = os.getenv("LLM_API_KEY", "not-needed")

print(f"--- INFO: 正在设置 OPENAI_API_BASE 环境变量为: '{LLM_BASE_URL_FOR_ENV}' ---")
os.environ["OPENAI_API_BASE"] = LLM_BASE_URL_FOR_ENV

print(f"--- INFO: 正在设置 OPENAI_API_KEY 环境变量为: '{LLM_API_KEY_FOR_ENV}' ---")
os.environ["OPENAI_API_KEY"] = LLM_API_KEY_FOR_ENV

# --- LLM 配置 ---
# !!! 关键：模型名称必须带有 'openai/' 前缀给 LiteLLM !!!
LLM_MODEL_FOR_LITELLM_PROVIDER_ID = os.getenv("LLM_MODEL", "openai//data/sj/models/Qwen3-8B") 
# ChatOpenAI 实例的参数
LLM_BASE_URL_FOR_CHATOPENAI = LLM_BASE_URL_FOR_ENV
LLM_API_KEY_FOR_CHATOPENAI = LLM_API_KEY_FOR_ENV

# 打印调试信息
_actual_model_path_for_vllm = LLM_MODEL_FOR_LITELLM_PROVIDER_ID.split('openai/', 1)[-1] if LLM_MODEL_FOR_LITELLM_PROVIDER_ID.startswith("openai/") else LLM_MODEL_FOR_LITELLM_PROVIDER_ID
print(f"--- DEBUG: LLM_MODEL environment variable = {os.getenv('LLM_MODEL')} ( defaulting to: {LLM_MODEL_FOR_LITELLM_PROVIDER_ID} for LiteLLM provider ID) ---")
print(f"--- DEBUG: (Actual model path expected by vLLM server if LiteLLM strips prefix: {_actual_model_path_for_vllm}) ---")
print(f"--- DEBUG: LLM_BASE_URL used for ChatOpenAI = {LLM_BASE_URL_FOR_CHATOPENAI} ---")
print(f"--- DEBUG: LLM_API_KEY used for ChatOpenAI = '{LLM_API_KEY_FOR_CHATOPENAI}' ---")
print(f"--- DEBUG: OPENAI_API_BASE environment variable is set to: '{os.environ.get('OPENAI_API_BASE')}' ---")
print(f"--- DEBUG: OPENAI_API_KEY environment variable is set to: '{os.environ.get('OPENAI_API_KEY')}' ---")

# --- 使用 LangChain 实例化 LLM ---
llm = None
try:
    # !!! 关键：使用带前缀的模型名称初始化 ChatOpenAI !!!
    # 这是为了让 CrewAI -> LiteLLM 能识别 provider
    print(f"--- 正在初始化 LangChain ChatOpenAI LLM (连接到 {LLM_BASE_URL_FOR_CHATOPENAI}, 模型 {LLM_MODEL_FOR_LITELLM_PROVIDER_ID}) ---")
    llm = ChatOpenAI(
        model=LLM_MODEL_FOR_LITELLM_PROVIDER_ID, # <-- 使用带前缀的名称
        temperature=0.1,
        openai_api_base=LLM_BASE_URL_FOR_CHATOPENAI,
        openai_api_key=LLM_API_KEY_FOR_CHATOPENAI,
        request_timeout=60
    )
    print("✅ LangChain ChatOpenAI LLM 初始化成功!")
    print(f"   实例模型名称 (llm.model_name): {getattr(llm, 'model_name', 'N/A')}") # 这里会显示 openai//data/sj/deepseek
    print(f"   实例API Base (llm.openai_api_base): {getattr(llm, 'openai_api_base', 'N/A')}")

except Exception as e:
    # ... (错误处理代码保持不变) ...
    print(f"❌ 初始化 LangChain ChatOpenAI LLM 时出错: {e}")
    print(f"   错误类型: {type(e).__name__}")
    print("   请检查:")
    print(f"     1. vLLM API 服务 ({LLM_BASE_URL_FOR_CHATOPENAI}) 是否正在运行且可访问?")
    print(f"     2. 环境变量 OPENAI_API_BASE 和 OPENAI_API_KEY 是否已正确设置?")
    print(f"     3. 模型名称 '{LLM_MODEL_FOR_LITELLM_PROVIDER_ID}' 是否适合 LiteLLM 识别 provider?")
    print("详细错误追踪信息:")
    traceback.print_exc()
    print("⚠️ LLM 实例未能创建，后续 CrewAI 工作流将无法运行。")


# --- 打印最终配置概览 ---
# (这部分代码不需要修改)
print("-" * 40)
print("LLM 配置加载完成 (使用 LangChain)")
if llm:
    model_used_in_instance = getattr(llm, 'model_name', 'N/A') # 获取实例中的模型名
    base_url_used_in_instance = getattr(llm, 'openai_api_base', 'N/A')
    api_key_in_env = os.environ.get("OPENAI_API_KEY", "未设置")
    api_base_in_env = os.environ.get("OPENAI_API_BASE", "未设置")
    env_var_status_desc = f"环境变量 OPENAI_API_KEY='{api_key_in_env}', OPENAI_API_BASE='{api_base_in_env}'"

    print(f"  LLM实例模型名 (llm.model_name): {model_used_in_instance}")
    print(f"  LLM实例基础URL (llm.openai_api_base): {base_url_used_in_instance}")
    print(f"  环境变量状态: {env_var_status_desc}")
    print("  状态: 初始化成功")
else:
    # ... (错误处理代码保持不变) ...
    model_to_print = LLM_MODEL_FOR_LITELLM_PROVIDER_ID
    print(f"  尝试使用模型 (Model for LiteLLM): {model_to_print} (尝试配置)")
    print(f"  基础 URL (Base URL): {LLM_BASE_URL_FOR_CHATOPENAI} (尝试配置)")
    api_key_in_env = os.environ.get("OPENAI_API_KEY", "未设置")
    api_base_in_env = os.environ.get("OPENAI_API_BASE", "未设置")
    env_var_status_desc = f"环境变量 OPENAI_API_KEY='{api_key_in_env}', OPENAI_API_BASE='{api_base_in_env}'"
    print(f"  环境变量状态: {env_var_status_desc}")
    print("  状态: 初始化失败 ❌")
print("-" * 40)

# --- 文件结束 ---