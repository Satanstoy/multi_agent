# tools/legal_tools.py
import os
from crewai.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from duckduckgo_search import DDGS

# --- RAG 工具配置 ---
PERSIST_DIRECTORY = "legal_db"  # 必须与 index_legal_docs.py 中使用的持久化路径相同
EMBEDDING_MODEL = "moka-ai/m3e-base"  # 使用的嵌入模型名称 (必须与索引时相同)
# --- 配置结束 ---

# --- 全局变量，用于延迟加载 RAG 组件 ---
vector_store = None
embeddings = None

def _initialize_rag():
    """如果 RAG 组件（向量存储和嵌入模型）尚未初始化，则进行初始化。"""
    global vector_store, embeddings
    if vector_store is None:
        print(f"--- [RAG 初始化] 正在从: {PERSIST_DIRECTORY} 加载向量存储 ---")
        if not os.path.exists(PERSIST_DIRECTORY):
            raise FileNotFoundError(f"错误：在 '{PERSIST_DIRECTORY}' 未找到向量存储，请先运行 'index_legal_docs.py' 创建。")

        print(f"--- [RAG 初始化] 正在初始化嵌入模型: {EMBEDDING_MODEL} ---")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("--- [RAG 初始化] 向量存储和嵌入模型加载成功 ---")

# --- 工具定义区 ---

@tool("类案匹配工具 (Similar Case Matching Tool) - SCM")
def similar_case_matching(query: str) -> str:
    """根据案情查找相似案例。"""
    print(f"--- [工具调用] 类案匹配工具 (SCM) ---")
    # 【占位实现】
    result = f"C_1, C_2"  # 返回案例ID
    return f"<SCM>{result}</EndSCM>"

@tool("法律条款检索工具 (Legal Article Search Tool) - LAS")
def legal_article_search_rag(query: str) -> str:
    """在本地法律知识库中检索相关法律条款。"""
    print(f"--- [工具调用] 法律条款检索工具 (LAS) ---")
    try:
        _initialize_rag()
        if vector_store is None:
            return "<LAS>错误：无法访问法律知识库</EndLAS>"

        results = vector_store.similarity_search(query, k=3)
        if not results:
            return f"<LAS>未找到相关法律条款</EndLAS>"

        formatted = []
        for doc in results:
            preview = doc.page_content.replace('\n', ' ').strip()[:200]
            formatted.append(preview)

        final_result = " ".join(formatted)
        return f"<LAS>{final_result}</EndLAS>"

    except Exception as e:
        return f"<LAS>检索失败: {e}</EndLAS>"

@tool("罪名预测工具 (Legal Charge Prediction Tool) - LCP")
def legal_charge_prediction(case_details: str) -> str:
    """根据案情预测可能罪名。"""
    print(f"--- [工具调用] 罪名预测工具 (LCP) ---")
    # 【占位实现】
    result = "盗窃罪, 抢夺罪"  # 示例预测
    return f"<LCP>{result}</EndLCP>"

@tool("法律要素识别工具 (Legal Element Recognition Tool) - LER")
def legal_element_recognition(query: str) -> str:
    """识别案情中的关键法律要素。"""
    print(f"--- [工具调用] 法律要素识别工具 (LER) ---")
    # 【占位实现】
    result = "主体: 被告人李某；行为: 拆解盗窃；结果: 财物损失"
    return f"<LER>{result}</EndLER>"

@tool("法律事件检测工具 (Legal Event Detection Tool) - LED")
def legal_event_detection(query: str) -> str:
    """检测案情描述中涉及的法律事件。"""
    print(f"--- [工具调用] 法律事件检测工具 (LED) ---")
    # 【占位实现】
    result = "事件：离婚诉讼, 财产分割"
    return f"<LED>{result}</EndLED>"

@tool("法律摘要工具 (Legal Text Summarization Tool) - LTS")
def legal_text_summary(query: str) -> str:
    """对复杂法律文本进行摘要。"""
    print(f"--- [工具调用] 法律摘要工具 (LTS) ---")
    # 【占位实现】
    result = "案情概括：双方因财产分割产生纠纷，提起诉讼。"
    return f"<LTS>{result}</EndLTS>"

@tool("互联网搜索工具 (Web Search Tool) - WEB")
def web_search(query: str) -> str:
    """通过DuckDuckGo搜索获取最新法律相关信息。"""
    print(f"--- [工具调用] 互联网搜索工具 (WEB) ---")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            search_results = list(results)

        if not search_results:
            return f"<WEB>未找到相关互联网信息</EndWEB>"

        formatted = []
        for r in search_results:
            title = r.get('title', '无标题')
            snippet = r.get('body', '无摘要')
            url = r.get('href', '')
            formatted.append(f"{title}: {snippet} ({url})")

        final_result = " ".join(formatted)
        return f"<WEB>{final_result}</EndWEB>"

    except Exception as e:
        return f"<WEB>搜索失败: {e}</EndWEB>"

available_tools = [
    similar_case_matching,
    legal_article_search_rag,
    legal_charge_prediction,
    legal_element_recognition,
    legal_event_detection,
    legal_text_summary,
    web_search
]