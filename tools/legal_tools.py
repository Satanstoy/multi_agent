# tools/legal_tools.py
import os
from crewai.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from duckduckgo_search import DDGS

# --- RAG 工具配置 ---
PERSIST_DIRECTORY = "/data/sj/multi_agent/legal_docs/legal_db"  # 必须与索引时使用的持久化路径相同
EMBEDDING_MODEL = "/data/sj/models/m3e-base"  # 使用的嵌入模型名称 (必须与索引时相同)
# --- 配置结束 ---

# --- 全局变量，用于延迟加载 RAG 组件 ---
vector_store = None
embeddings = None

def _initialize_rag():
    """如果 RAG 组件尚未初始化，则进行初始化。"""
    global vector_store, embeddings
    if vector_store is None:
        print(f"--- [RAG 初始化] 尝试从: {PERSIST_DIRECTORY} 加载向量存储 ---")
        if not os.path.exists(PERSIST_DIRECTORY):
            # 修改：如果数据库不存在，仅打印警告并返回，不抛出错误中断程序
            print(f"⚠️ 警告：在 '{PERSIST_DIRECTORY}' 未找到向量存储。RAG 工具将不可用。请运行索引脚本创建数据库。")
            return
            # raise FileNotFoundError(f"错误：在 '{PERSIST_DIRECTORY}' 未找到向量存储，请先运行 'index_legal_docs.py' 创建。")

        try:
            print(f"--- [RAG 初始化] 正在初始化嵌入模型: {EMBEDDING_MODEL} ---")
            # 注意：根据你的环境，可能需要调整 device='cpu' 或 device='cuda'
            embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
            vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
            print("--- [RAG 初始化] 向量存储和嵌入模型加载成功 ---")
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载向量存储或嵌入模型时出错: {e}")
            vector_store = None # 确保出错时 vector_store 为 None

# --- 工具定义区 (已精简名称和描述) ---

@tool("相似案例查找(SCM)")
def similar_case_matching(query: str) -> str:
    """输入案情(query: str)，查找相似案例。"""
    print(f"--- [工具调用] 相似案例查找(SCM) ---")
    # 【占位实现】
    # 实际应用中，这里应调用你的类案匹配模型或服务
    # 例如: response = call_scm_service(query)
    # result = f"相关案例ID: {response.get('case_ids', '无')}"
    result = f"C_1, C_2" # 返回案例ID (示例)
    return f"<SCM>{result}</EndSCM>"

@tool("法条检索(LAS)")
def legal_article_search_rag(query: str) -> str:
    """输入问题(query: str)，在本地库检索相关法条。"""
    print(f"--- [工具调用] 法条检索(LAS) ---")
    try:
        _initialize_rag() # 尝试初始化 RAG
        if vector_store is None:
            # 如果初始化失败或未找到数据库，返回明确信息
            return "<LAS>错误：无法访问本地法律知识库</EndLAS>"

        # 执行相似性搜索
        results = vector_store.similarity_search(query, k=2) # 减少返回数量，节省 token
        if not results:
            return f"<LAS>未找到相关法律条款</EndLAS>"

        # 格式化结果
        formatted = []
        for i, doc in enumerate(results):
            # 进一步缩短预览长度
            preview = doc.page_content.replace('\n', ' ').strip()[:150]
            formatted.append(f"法条片段{i+1}: {preview}...") # 添加编号和省略号

        final_result = " ".join(formatted)
        return f"<LAS>{final_result}</EndLAS>"

    except Exception as e:
        print(f"❌ [LAS 工具错误] 检索时发生错误: {e}") # 添加错误日志
        return f"<LAS>检索法条失败</EndLAS>" # 返回更简洁的错误信息

@tool("罪名预测(LCP)")
def legal_charge_prediction(case_details: str) -> str:
    """输入案情(case_details: str)，预测可能罪名。"""
    print(f"--- [工具调用] 罪名预测(LCP) ---")
    # 【占位实现】
    # 实际应用中调用预测模型
    result = "盗窃罪, 抢夺罪"
    return f"<LCP>{result}</EndLCP>"

@tool("法律要素识别(LER)")
def legal_element_recognition(query: str) -> str:
    """输入案情(query: str)，识别关键法律要素。"""
    print(f"--- [工具调用] 法律要素识别(LER) ---")
    # 【占位实现】
    # 实际应用中调用识别模型
    result = "主体: 李某；行为: 拆解盗窃；结果: 财物损失"
    return f"<LER>{result}</EndLER>"

@tool("法律事件检测(LED)")
def legal_event_detection(query: str) -> str:
    """输入案情(query: str)，检测涉及的法律事件。"""
    print(f"--- [工具调用] 法律事件检测(LED) ---")
    # 【占位实现】
    # 实际应用中调用检测模型
    result = "事件：离婚诉讼, 财产分割"
    return f"<LED>{result}</EndLED>"

@tool("法律文本摘要(LTS)")
def legal_text_summary(query: str) -> str:
    """输入法律文本(query: str)，生成摘要。"""
    print(f"--- [工具调用] 法律文本摘要(LTS) ---")
    # 【占位实现】
    # 实际应用中调用摘要模型
    result = "案情概括：双方因财产分割产生纠纷，提起诉讼。"
    return f"<LTS>{result}</EndLTS>"

@tool("互联网搜索(WEB)")
def web_search(query: str) -> str:
    """输入问题(query: str)，用DuckDuckGo搜索网络信息。"""
    print(f"--- [工具调用] 互联网搜索(WEB) ---")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2) # 减少结果数量
            search_results = list(results)

        if not search_results:
            return f"<WEB>未找到相关互联网信息</EndWEB>"

        # 格式化结果，使其更简洁
        formatted = []
        for r in search_results:
            title = r.get('title', '')
            snippet = r.get('body', '')[:100] # 缩短摘要长度
            formatted.append(f"{title}: {snippet}...")

        final_result = " ".join(formatted)
        return f"<WEB>{final_result}</EndWEB>"

    except Exception as e:
        print(f"❌ [WEB 工具错误] 搜索时发生错误: {e}") # 添加错误日志
        return f"<WEB>网络搜索失败</EndWEB>" # 返回更简洁的错误信息

# 这个列表包含了上面定义的所有工具函数
available_tools = [
    similar_case_matching,
    legal_article_search_rag,
    legal_charge_prediction,
    legal_element_recognition,
    legal_event_detection,
    legal_text_summary,
    web_search
]

# 可以在这里加一行打印，确认工具名称已被缩短
print(f"--- [工具加载] 可用工具列表 (已加载): {[tool.name for tool in available_tools]} ---")