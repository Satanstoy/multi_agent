# multi_agent/tools/legal_tools.py

import os
import torch
import traceback
from langchain_core.messages import HumanMessage, SystemMessage
# 关键：让工具直接使用共享的 llm 实例，并使用 @tool 装饰器
from config import llm 
from crewai.tools import tool

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from duckduckgo_search import DDGS

# --- 路径和初始化函数部分 (保持不变) ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
EMBEDDING_MODEL_PATH = "/data/sj/models/m3e-base"
LEGAL_DB_PATH = os.path.join(project_root, "docs", "legal_db")
CASE_DB_PATH = os.path.join(project_root, "docs", "case_db")

embeddings = None
legal_vector_store = None
case_vector_store = None

def _initialize_embeddings():
    """如果嵌入模型尚未初始化，则进行初始化并设为全局变量。"""
    global embeddings
    if embeddings is None:
        print(f"--- [RAG 初始化] 首次加载嵌入模型: {EMBEDDING_MODEL_PATH} ---")
        if not os.path.exists(EMBEDDING_MODEL_PATH):
            print(f"⚠️ 警告：在 '{EMBEDDING_MODEL_PATH}' 未找到嵌入模型。RAG 工具将不可用。")
            return
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- [RAG 初始化] 自动检测到可用设备: {device.upper()} ---")

        try:
            embeddings = SentenceTransformerEmbeddings(
                model_name=EMBEDDING_MODEL_PATH,
                model_kwargs={'device': device}
            )
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载嵌入模型时出错: {e}")
            embeddings = None

def _initialize_legal_rag():
    """初始化法条 RAG 组件。"""
    global legal_vector_store
    if legal_vector_store is None:
        _initialize_embeddings()
        if embeddings is None or not os.path.exists(LEGAL_DB_PATH): return
        try:
            legal_vector_store = Chroma(persist_directory=LEGAL_DB_PATH, embedding_function=embeddings)
            print("--- [RAG 初始化] 法条向量存储加载成功 ---")
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载法条向量存储时出错: {e}")

def _initialize_case_rag():
    """初始化案例 RAG 组件。"""
    global case_vector_store
    if case_vector_store is None:
        _initialize_embeddings()
        if embeddings is None or not os.path.exists(CASE_DB_PATH): return
        try:
            case_vector_store = Chroma(persist_directory=CASE_DB_PATH, embedding_function=embeddings)
            print("--- [RAG 初始化] 案例向量存储加载成功 ---")
        except Exception as e:
            print(f"❌ [RAG 初始化] 加载案例向量存储时出错: {e}")

# --- 工具定义区 ---

@tool("相似案例查找(SCM)")
def similar_case_matching(query: str, k: int = 3) -> str:
    """
    当需要寻找与当前案件相似的先例时使用此工具。
    输入参数 'query' 应该是一段详细的案情描述，至少包含案件的关键事实、争议焦点等信息。
    此工具会从本地的判例数据库中，找出语义上最接近的案例，并返回它们的来源、内容预览和相关性分数。
    例如：'被告人李四于2024年5月晚间，撬开被害人王五家门，窃取了价值五千元的笔记本电脑一台'。
    """
    print(f"--- [工具调用] 相似案例查找(SCM) | 检索数量: {k} ---")
    try:
        _initialize_case_rag()
        if case_vector_store is None:
            return "<SCM status='error'>无法访问本地案例知识库。请确认已成功运行索引脚本创建case_db。</SCM>"

        results = case_vector_store.similarity_search_with_relevance_scores(query, k=k)
        if not results:
            return f"<SCM status='not_found'>未在案例库中找到与您描述相似的案例。</SCM>"

        formatted = []
        for i, (doc, score) in enumerate(results):
            source = os.path.basename(doc.metadata.get('source', '未知来源'))
            preview = doc.page_content.replace('\n', ' ').strip()[:150]
            formatted.append(f"相似案例{i+1}(来源:{source}, 相关性得分:{score:.4f}): {preview}...")
        
        final_result = " | ".join(formatted)
        return f"<SCM status='success'>检索到以下案例（相关性得分越低表示越相似）: {final_result}</SCM>"
    except Exception as e:
        print(f"❌ [SCM 工具错误] 检索时发生错误: {e}\n{traceback.format_exc()}")
        return f"<SCM status='error'>系统在检索相似案例时发生内部错误: {e}</SCM>"


@tool("法条检索(LAS)")
def legal_article_search_rag(query: str, k: int = 3, fetch_k: int = 10) -> str:
    """
    当需要查找、引用或验证相关法律条款时使用此工具。
    输入 'query' 可以是案情描述或直接的法律问题。
    此工具会利用MMR(最大边际相关性)算法从法条库中检索多样化且相关的法律条文。
    可以指定 'k' 来控制返回的法条数量。
    """
    print(f"--- [工具调用] 法条检索(LAS) | 检索数量: {k}, MMR候选: {fetch_k} ---")
    try:
        _initialize_legal_rag() 
        if legal_vector_store is None:
            return "<LAS status='error'>错误：无法访问本地法律知识库。</LAS>"

        retriever = legal_vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': k, 'fetch_k': fetch_k}
        )
        results = retriever.get_relevant_documents(query)
        
        if not results:
            return f"<LAS status='not_found'>未在法条库中找到与 '{query}' 相关的法律条款。</LAS>"

        formatted = []
        for i, doc in enumerate(results):
            source = os.path.basename(doc.metadata.get('source', '未知来源'))
            preview = doc.page_content.replace('\n', ' ').strip()
            formatted.append(f"法条片段{i+1}(来源:{source}): {preview}")

        final_result = " | ".join(formatted)
        return f"<LAS status='success'>{final_result}</LAS>"
    except Exception as e:
        print(f"❌ [LAS 工具错误] 检索时发生错误: {e}\n{traceback.format_exc()}")
        return f"<LAS status='error'>检索法条时发生内部错误: {e}</LAS>"

@tool("互联网搜索(WEB)")
def web_search(query: str) -> str:
    """
    当需要获取最新的、非本地知识库包含的公开信息时使用此工具。
    例如查询最新的法律修正案、特定人物的背景信息或时事新闻。
    输入一个清晰的搜索问题 'query'。
    """
    print(f"--- [工具调用] 互联网搜索(WEB) ---")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return f"<WEB status='not_found'>未找到与 '{query}' 相关的互联网信息。</WEB>"
        
        formatted = []
        for r in results:
            title = r.get('title', '')
            snippet = r.get('body', '')
            link = r.get('href', '')
            formatted.append(f"标题: {title}\n摘要: {snippet}\n链接: {link}")
            
        final_result = "\n---\n".join(formatted)
        return f"<WEB status='success'>\n{final_result}\n</WEB>"
    except Exception as e:
        print(f"❌ [WEB 工具错误] 搜索时发生错误: {e}")
        return f"<WEB status='error'>网络搜索失败: {e}</WEB>"

@tool("罪名预测(LCP)")
def legal_charge_prediction(case_details: str) -> str:
    """
    输入一个详细的案情描述(case_details)，此工具会执行一个完整的RAG流程来预测最可能的罪名。
    流程包括：1.在刑法库中检索相关法条。2.结合案情和法条，调用大语言模型进行推理。3.返回最终的罪名。
    当用户想知道某个行为构成什么罪时，应使用此工具。
    """
    print(f"--- [工具调用] 罪名预测(LCP) - 完整的RAG流程启动 ---")
    retrieved_articles = ""
    try:
        _initialize_legal_rag()
        if legal_vector_store:
            results = legal_vector_store.similarity_search(case_details, k=3)
            if results:
                formatted = [f"相关法条片段{i+1}: {doc.page_content.strip()}" for i, doc in enumerate(results)]
                retrieved_articles = "\n\n".join(formatted)
    except Exception as e:
        retrieved_articles = f"内部检索法条时发生错误: {e}"

    prompt = f"""
    作为一名资深的中国刑事法律专家，请严格根据以下信息进行分析。
    [案情描述]: {case_details}
    [从法条库检索到的最相关法律规定 (仅供你参考，要以案情为准)]: {retrieved_articles or '无'}
    [你的任务]: 请你综合分析上述[案情描述]，并参考[相关法律规定]，准确地判断并给出该案情最可能构成的一个或多个具体罪名。你的回答应该非常简洁，请【只返回罪名名称本身】，如果有多个，请用逗号分隔。如果信息不足以做出明确判断，请返回 '根据现有信息无法准确判断罪名'。
    """
    try:
        response = llm.invoke(prompt)
        final_charge = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return f"<LCP status='success'>{final_charge}</LCP>"
    except Exception as e:
        return f"<LCP status='error'>在进行罪名推理时发生内部错误: {e}</LCP>"

@tool("法律要素识别(LER)")
def legal_element_recognition(query: str) -> str:
    """
    用于从一段详细的案情描述中，抽取出结构化的法律核心要素。
    输入 'query' 应该是案情描述。
    返回案件的关键要素，如主体、客体、客观方面、主观方面等。
    当需要对案情进行结构化分析时使用此工具。
    """
    prompt = f"""
    作为一名精通中国法律的AI分析师，你的任务是从以下[案情描述]中，抽取出犯罪构成的四个核心法律要件。
    [案情描述]: {query}
    [你的任务]: 请严格按照下面的格式进行输出，并对每个要件进行简洁的描述。如果某个要素在文本中不明确或不存在，请填写'不明确'。
    主体: [此处填写犯罪主体]
    客体: [此处填写犯罪行为所侵犯的社会关系]
    客观方面: [此处填写犯罪行为的具体表现]
    主观方面: [此处填写行为人的心理状态]
    """
    try:
        response = llm.invoke(prompt)
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return f"<LER status='success'>{result}</LER>"
    except Exception as e:
        return f"<LER status='error'>在进行法律要素识别时发生内部错误: {e}</LER>"

@tool("法律事件检测(LED)")
def legal_event_detection(query: str) -> str:
    """

    用于从一段复杂的文本（如案情描述、用户对话）中，检测并列出所有涉及到的具体法律事件或程序。
    输入 'query' 是包含事件的文本。
    返回一个由逗号分隔的事件列表。
    """
    prompt = f"""
    作为AI法律事件检测器，你的任务是从下面提供的[文本]中，识别并列出所有具体的法律事件或法律程序。
    [文本]: {query}
    [任务说明]: 你需要检测的事件类型包括但不限于：'提起诉讼', '申请仲裁', '签订合同', '提出上诉', '离婚登记', '财产分割', '工伤认定', '申请强制执行', '继承遗产', '报案' 等。请将所有识别出的事件用逗号分隔，并只输出事件名称本身。如果未检测到任何明确的法律事件，请返回'未检测到特定法律事件'。
    """
    try:
        response = llm.invoke(prompt)
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return f"<LED status='success'>{result}</LED>"
    except Exception as e:
        return f"<LED status='error'>在进行法律事件检测时发生内部错误: {e}</LER>"

@tool("法律文本摘要(LTS)")
def legal_text_summary(query: str) -> str:
    """
    用于将长篇的法律文书、案情描述或任何法律相关文本，生成一段简洁、中立、准确的摘要。
    输入 'query' 是需要被摘要的长文本。
    返回一段概括性的摘要文本。
    """
    prompt = f"""
    作为一名专业的AI法律文书摘要师，请将以下[原始法律文本]内容，提炼成一段不超过200字的、简洁、准确、中立的摘要。摘要需要突出核心事实、主要人物关系和关键的争议焦点或结论。
    [原始法律文本]: {query}
    [输出要求]: 请直接输出摘要内容，不要添加“摘要如下：”等多余的引言。
    """
    try:
        response = llm.invoke(prompt)
        result = response.content.strip() if hasattr(response, 'content') else str(response).strip()
        return f"<LTS status='success'>{result}</LTS>"
    except Exception as e:
        return f"<LTS status='error'>在生成法律文本摘要时发生内部错误: {e}</LTS>"


# --- 工具列表 ---
available_tools = [         
    similar_case_matching,
    legal_article_search_rag,
    legal_charge_prediction,
    legal_element_recognition,
    legal_event_detection,
    legal_text_summary,
    web_search
]