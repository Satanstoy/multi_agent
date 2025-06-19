import os
from crewai.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from duckduckgo_search import DDGS

# --- 新增/修改：健壮的路径配置 ---
# 工具文件位于 multi_agent/tools/
script_dir = os.path.dirname(os.path.abspath(__file__))  # .../tools
project_root = os.path.dirname(script_dir)              # .../multi_agent

# 计算模型和数据库的绝对路径，使其在任何位置运行都有效
EMBEDDING_MODEL_PATH = os.path.join(project_root, "models", "m3e-base")
LEGAL_DB_PATH = os.path.join(project_root, "docs", "legal_db")
CASE_DB_PATH = os.path.join(project_root, "docs", "case_db")
# --- 配置修改结束 ---


# --- 新增/修改：全局变量，用于延迟加载和共享 RAG 组件 ---
embeddings = None # 全局共享的嵌入模型
legal_vector_store = None # 法条数据库实例
case_vector_store = None  # 案例数据库实例

def _initialize_embeddings():
    """如果嵌入模型尚未初始化，则进行初始化并设为全局变量。"""
    global embeddings
    if embeddings is None:
        print(f"--- [RAG 初始化] 首次加载嵌入模型: {EMBEDDING_MODEL_PATH} ---")
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            print(f"⚠️ 警告：在 '{EMBEDDING_MODEL_PATH}' 未找到嵌入模型。RAG 工具将不可用。")
            return
        try:
            embeddings = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载嵌入模型时出错: {e}")
            embeddings = None # 加载失败

def _initialize_legal_rag():
    """初始化法条 RAG 组件。"""
    global legal_vector_store
    if legal_vector_store is None:
        _initialize_embeddings() # 确保嵌入模型已加载
        if embeddings is None: return # 如果嵌入失败则退出

        print(f"--- [RAG 初始化] 尝试从: {LEGAL_DB_PATH} 加载法条向量存储 ---")
        if not os.path.exists(LEGAL_DB_PATH):
            print(f"⚠️ 警告：在 '{LEGAL_DB_PATH}' 未找到法条库。请先运行索引脚本创建。")
            return
        try:
            legal_vector_store = Chroma(persist_directory=LEGAL_DB_PATH, embedding_function=embeddings)
            print("--- [RAG 初始化] 法条向量存储加载成功 ---")
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载法条向量存储时出错: {e}")

def _initialize_case_rag():
    """初始化案例 RAG 组件。"""
    global case_vector_store
    if case_vector_store is None:
        _initialize_embeddings() # 确保嵌入模型已加载
        if embeddings is None: return # 如果嵌入失败则退出

        print(f"--- [RAG 初始化] 尝试从: {CASE_DB_PATH} 加载案例向量存储 ---")
        if not os.path.exists(CASE_DB_PATH):
            print(f"⚠️ 警告：在 '{CASE_DB_PATH}' 未找到案例库。请先运行索引脚本创建。")
            return
        try:
            case_vector_store = Chroma(persist_directory=CASE_DB_PATH, embedding_function=embeddings)
            print("--- [RAG 初始化] 案例向量存储加载成功 ---")
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载案例向量存储时出错: {e}")
# --- 新增/修改结束 ---


# --- 工具定义区 ---
@tool("相似案例查找(SCM)")
def similar_case_matching(query: str) -> str:
    """输入案情(query: str)，在本地案例库中检索语义最相似的案例。"""
    print(f"--- [工具调用] 相似案例查找(SCM) ---")
    try:
        # 调用专为案例库设计的初始化函数
        _initialize_case_rag()
        if case_vector_store is None:
            return "<SCM>错误：无法访问本地案例知识库。请确认已成功运行索引脚本创建case_db。</EndSCM>"

        # 从案例数据库执行相似性搜索
        results = case_vector_store.similarity_search(query, k=3) # 查找3个最相似的案例
        if not results:
            return f"<SCM>未在案例库中找到与您描述相似的案例。</EndSCM>"

        # 格式化输出，使其包含更多有用信息
        formatted = []
        for i, doc in enumerate(results):
            # 从metadata中获取源文件名，如果不存在则使用'未知来源'
            source = os.path.basename(doc.metadata.get('source', '未知来源'))
            # 提取案情内容预览
            preview = doc.page_content.replace('\n', ' ').strip()[:150]
            # 格式化每个结果
            formatted.append(f"相似案例{i+1}(来源:{source}): {preview}...")
        
        final_result = " | ".join(formatted) # 使用 | 分隔符，让LLM更容易解析
        return f"<SCM>{final_result}</EndSCM>"
    except Exception as e:
        print(f"❌ [SCM 工具错误] 检索时发生错误: {e}")
        return f"<SCM>系统在检索相似案例时发生内部错误。</EndSCM>"
# --- SCM工具修改结束 ---

@tool("法条检索(LAS)")
def legal_article_search_rag(query: str) -> str:
    """输入问题(query: str)，在本地法条库检索相关法条。"""
    print(f"--- [工具调用] 法条检索(LAS) ---")
    try:
        # --- 修改：调用新的法条初始化函数 ---
        _initialize_legal_rag() 
        if legal_vector_store is None:
            return "<LAS>错误：无法访问本地法律知识库。</EndLAS>"

        # --- 修改：从 legal_vector_store 检索 ---
        results = legal_vector_store.similarity_search(query, k=2) 
        if not results:
            return f"<LAS>未找到相关法律条款</EndLAS>"

        # 格式化结果 (逻辑不变)
        formatted = []
        for i, doc in enumerate(results):
            preview = doc.page_content.replace('\n', ' ').strip()[:150]
            formatted.append(f"法条片段{i+1}: {preview}...")

        final_result = " ".join(formatted)
        return f"<LAS>{final_result}</EndLAS>"
    except Exception as e:
        print(f"❌ [LAS 工具错误] 检索时发生错误: {e}")
        return f"<LAS>检索法条失败</EndLAS>"

# --- 其他占位符工具保持不变 ---

@tool("罪名预测(LCP)")
def legal_charge_prediction(case_details: str) -> str:
    """输入案情(case_details: str)，预测可能罪名。"""
    print(f"--- [工具调用] 罪名预测(LCP) ---")
    result = "盗窃罪, 抢夺罪"
    return f"<LCP>{result}</EndLCP>"

@tool("法律要素识别(LER)")
def legal_element_recognition(query: str) -> str:
    """输入案情(query: str)，识别关键法律要素。"""
    print(f"--- [工具调用] 法律要素识别(LER) ---")
    result = "主体: 李某；行为: 拆解盗窃；结果: 财物损失"
    return f"<LER>{result}</EndLER>"

@tool("法律事件检测(LED)")
def legal_event_detection(query: str) -> str:
    """输入案情(query: str)，检测涉及的法律事件。"""
    print(f"--- [工具调用] 法律事件检测(LED) ---")
    result = "事件：离婚诉讼, 财产分割"
    return f"<LED>{result}</EndLED>"

@tool("法律文本摘要(LTS)")
def legal_text_summary(query: str) -> str:
    """输入法律文本(query: str)，生成摘要。"""
    print(f"--- [工具调用] 法律文本摘要(LTS) ---")
    result = "案情概括：双方因财产分割产生纠纷，提起诉讼。"
    return f"<LTS>{result}</EndLTS>"

@tool("互联网搜索(WEB)")
def web_search(query: str) -> str:
    """输入问题(query: str)，用DuckDuckGo搜索网络信息。"""
    print(f"--- [工具调用] 互联网搜索(WEB) ---")
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=2)
            search_results = list(results)
        if not search_results:
            return f"<WEB>未找到相关互联网信息</EndWEB>"
        formatted = []
        for r in search_results:
            title = r.get('title', '')
            snippet = r.get('body', '')[:100]
            formatted.append(f"{title}: {snippet}...")
        final_result = " ".join(formatted)
        return f"<WEB>{final_result}</EndWEB>"
    except Exception as e:
        print(f"❌ [WEB 工具错误] 搜索时发生错误: {e}")
        return f"<WEB>网络搜索失败</EndWEB>"

# --- 工具列表保持不变 ---
available_tools = [
    similar_case_matching,
    legal_article_search_rag,
    legal_charge_prediction,
    legal_element_recognition,
    legal_event_detection,
    legal_text_summary,
    web_search
]

print(f"--- [工具加载] 可用工具列表 (已加载): {[tool.name for tool in available_tools]} ---")