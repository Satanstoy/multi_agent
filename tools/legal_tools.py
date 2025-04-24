# tools/legal_tools.py
import os
from crewai import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- RAG 工具配置 ---
PERSIST_DIRECTORY = "legal_db" # 必须与 index_legal_docs.py 中使用的持久化路径相同
# 务必使用与索引时*完全相同*的嵌入模型
EMBEDDING_MODEL = "moka-ai/m3e-base" # 使用的嵌入模型名称 (必须与索引时相同)
# --- 配置结束 ---

# --- 用于保存加载的向量存储和嵌入模型的全局变量 ---
# 采用延迟加载策略：初始化为 None，在工具首次被调用时才加载
vector_store = None
embeddings = None

def _initialize_rag():
    """如果 RAG 组件尚未初始化，则进行初始化。"""
    global vector_store, embeddings # 声明我们要修改全局变量

    if vector_store is None: # 仅在首次需要时加载
        print(f"--- [RAG 初始化] 正在从以下位置加载向量存储: {PERSIST_DIRECTORY} ---")
        if not os.path.exists(PERSIST_DIRECTORY):
             # 如果索引数据库不存在，抛出错误，提示用户先运行索引脚本
             raise FileNotFoundError(f"错误：在 '{PERSIST_DIRECTORY}' 未找到向量存储。"
                                     "请先运行 'index_legal_docs.py' 来创建索引。")

        # 初始化嵌入模型 (必须与索引时使用的模型一致)
        # 如果有 GPU，可将 device='cpu' 改为 'cuda'
        print(f"--- [RAG 初始化] 正在初始化嵌入模型: {EMBEDDING_MODEL} ---")
        embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

        # 加载持久化的 Chroma 向量存储
        vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        print("--- [RAG 初始化] 向量存储和嵌入模型加载成功。 ---")
    # 理论上 embeddings 会和 vector_store 一起加载，这里可以省略二次检查
    # if embeddings is None:
    #     embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})


@tool("法律条款检索工具 (Legal Article Search Tool - RAG)") # 工具名称稍作修改以示区别
def legal_article_search_rag(query: str) -> str:
    """
    (RAG 版本) 根据用户查询（案情描述或法律问题）在本地法律知识库中检索相关的法律法规条款片段。
    输入: query (str) - 用户的问题或案情描述。
    返回: str - 从知识库中检索到的相关法条文本片段的集合。
    """
    print(f"--- [工具调用] 法律条款检索工具 (RAG) ---")
    print(f"    查询内容 (前50字符): {query[:50]}...")

    try:
        # 确保 RAG 组件（向量存储和嵌入模型）已加载
        _initialize_rag()

        if vector_store is None: # 双重检查，确保加载成功
             return "错误：无法访问法律知识库。请检查配置和索引文件是否就绪。"

        print(f"    正在执行相似性搜索...")
        # 在向量存储中执行相似性搜索，检索最相关的 N 个文档块 (k 值可调)
        results = vector_store.similarity_search(query, k=3) # 获取最相关的 3 个结果

        if not results:
            print("    未在知识库中找到相关文档。")
            return f"未能根据 '{query[:30]}...' 在知识库中找到直接相关的法律条款。"

        # 格式化检索结果以便阅读
        formatted_results = []
        print(f"    找到 {len(results)} 个相关片段。正在格式化...")
        for i, doc in enumerate(results):
            # 尝试获取源文件名（如果元数据中有）
            source = doc.metadata.get('source', '未知来源')
            # 获取文档块内容，做简单清理和预览
            content_preview = doc.page_content.replace('\n', ' ').strip()[:200] # 预览长度可调
            formatted_results.append(f"相关片段 {i+1} (来源: {source}):\n{content_preview}...\n---")

        # 构建最终返回给用户的字符串
        final_output = f"根据您的查询 '{query[:30]}...'，在法律知识库中找到以下相关内容：\n\n" + "\n".join(formatted_results)
        print(f"    工具返回内容预览 (前100字符): {final_output[:100]}...")
        return final_output

    except FileNotFoundError as fnf_error:
        print(f"    [错误] RAG 初始化失败: {fnf_error}")
        return str(fnf_error) # 将文件未找到的错误信息返回给 Agent
    except Exception as e:
        print(f"    [错误] RAG 检索过程中发生异常: {e}")
        # 在调试时可以取消下面两行的注释来打印详细的错误堆栈信息
        # import traceback
        # traceback.print_exc()
        return f"检索相关法律条款时遇到错误: {e}"

# --- 其他工具 (保持不变，但注释也翻译一下) ---
@tool("相似案例匹配工具 (Similar Case Matching Tool)")
def similar_case_matching(query: str) -> str:
    """
    根据用户查询（案情描述）查找相似的法律案例。
    输入: query (str) - 用户的问题或案情描述。
    返回: str - 相似案例的摘要、ID列表或相关信息。[占位符]
    """
    print(f"--- [工具调用] 相似案例匹配工具 ---")
    print(f"    查询内容 (前50字符): {query[:50]}...")
    # 【实际逻辑占位符】在这里调用你的类案检索模型或数据库查询
    result = f"找到与 '{query[:30]}...' 相关的相似案例：案例A (判决书ID: XXXXX), 案例B (判决书ID: YYYYY)。 [占位符]"
    print(f"    工具返回 (前50字符): {result[:50]}...")
    return result

@tool("罪名预测工具 (Legal Charge Prediction Tool)")
def legal_charge_prediction(case_details: str) -> str:
    """
    根据提供的案件细节预测可能涉及的罪名。
    输入: case_details (str) - 详细的案情描述。
    返回: str - 可能涉及的罪名列表及其置信度（可选）。[占位符]
    """
    print(f"--- [工具调用] 罪名预测工具 ---")
    print(f"    案情细节 (前50字符): {case_details[:50]}...")
    # 【实际逻辑占位符】在这里调用你的罪名预测模型
    result = f"基于案情 '{case_details[:30]}...'，预测可能涉及的罪名：盗窃罪 (较高可能性), 抢夺罪 (较低可能性)。 [占位符]"
    print(f"    工具返回 (前50字符): {result[:50]}...")
    return result

@tool("法律要素识别工具 (Legal Element Recognition Tool)")
def legal_element_recognition(case_details: str) -> str:
    """
    从案件细节中识别出关键的法律构成要素。
    输入: case_details (str) - 详细的案情描述。
    返回: str - 识别出的关键法律要素列表 (如主体、客体、主观方面、客观方面等)。[占位符]
    """
    print(f"--- [工具调用] 法律要素识别工具 ---")
    print(f"    案情细节 (前50字符): {case_details[:50]}...")
    # 【实际逻辑占位符】在这里调用你的要素识别模型
    result = f"在 '{case_details[:30]}...' 中识别出要素：主体-张三, 行为-秘密窃取, 对象-他人财物。 [占位符]"
    print(f"    工具返回 (前50字符): {result[:50]}...")
    return result

@tool("网络搜索工具 (Web Search Tool)")
def web_search(query: str) -> str:
    """
    执行互联网搜索以获取最新的信息、背景知识或特定问题的答案。
    输入: query (str) - 需要搜索的查询语句。
    返回: str - 搜索结果的摘要或关键信息。[占位符，需要替换为实际 API 调用]
    """
    print(f"--- [工具调用] 网络搜索工具 ---")
    print(f"    搜索查询: {query[:50]}...")
    # 【实际逻辑占位符】在这里集成真实的搜索引擎 API (如 Tavily, Google Search API, Serper)
    result = f"关于 '{query[:30]}...' 的网络搜索结果：...最新进展...相关定义... [占位符]"
    print(f"    工具返回 (前50字符): {result[:50]}...")
    return result

# --- 可用工具列表 ---
# 将所有定义好的工具实例收集到一个列表中，方便 Agent 使用
# 注意：这里将旧的占位符工具替换为新的 RAG 工具
available_tools = [
    similar_case_matching,
    legal_article_search_rag, # 使用我们新实现的 RAG 版本的法律条款检索工具
    legal_charge_prediction,
    legal_element_recognition,
    # legal_event_detection, # 如果实现了这个工具，也加进来
    # legal_text_summary,  # 如果实现了这个工具，也加进来
    web_search
]

# 打印加载的工具列表，方便调试
print("-" * 30)
print(f"工具模块加载完成，当前可用工具:")
for t in available_tools:
    print(f"  - {t.name}") # 打印每个工具的名称
print("-" * 30)