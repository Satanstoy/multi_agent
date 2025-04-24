# legal_docs/index_legal_docs.py
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- 配置 ---
SOURCE_DIRECTORY = "legal_docs"  # 包含你的法律文本文件的目录
PERSIST_DIRECTORY = "legal_db"  # 向量数据库将被保存到的路径
# 选择一个合适的嵌入模型 (多语言或中文侧重)
# 例如: 'moka-ai/m3e-base', 'shibing624/text2vec-base-chinese', 'BAAI/bge-large-zh-v1.5'
# 确保你选择的模型已经下载或可以通过库访问
EMBEDDING_MODEL = "moka-ai/m3e-base" # 使用的嵌入模型名称 (请根据实际情况选择)
CHUNK_SIZE = 500                 # 文本分割时每个块的目标大小
CHUNK_OVERLAP = 50               # 相邻块之间的重叠字符数
# --- 配置结束 ---

def create_vector_store():
    """加载文档、分割、嵌入并创建持久化的向量存储"""
    print(f"从目录加载文档: {SOURCE_DIRECTORY}")
    # 根据你的文件类型配置加载器 (TextLoader, PyPDFLoader 等)
    # 此处示例使用 TextLoader 加载 .txt 文件
    loader = DirectoryLoader(SOURCE_DIRECTORY, glob="**/*.txt", loader_cls=TextLoader, show_progress=True, use_multithreading=True)
    documents = loader.load() # 加载文档

    if not documents: # 检查是否成功加载了文档
        print(f"错误：在目录 '{SOURCE_DIRECTORY}' 中未找到任何 .txt 文档。请检查路径和文件。")
        return

    print(f"已加载 {len(documents)} 个文档。")

    print("正在将文档分割成块...")
    # 初始化递归字符文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents) # 分割文档
    print(f"已将文档分割成 {len(docs)} 个块。")

    print(f"正在初始化嵌入模型: {EMBEDDING_MODEL}")
    # 初始化句子转换器嵌入模型
    # 如果你有可用的 GPU 并且安装了支持 CUDA 的 PyTorch，可以将 device='cpu' 改为 device='cuda' 以加速
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={'device': 'cpu'})

    print(f"正在创建向量存储并将其持久化到: {PERSIST_DIRECTORY}")
    # 确保持久化目录存在
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # 从分割后的文档块创建 Chroma 向量存储
    # 这会将文档块进行嵌入并存入数据库
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY # 指定持久化路径
    )

    # 可选：显式调用持久化 (虽然 Chroma 在创建时通常已持久化)
    vector_store.persist()
    print("向量存储创建并持久化成功！")

if __name__ == "__main__":
    # 当直接运行此脚本时，执行创建向量存储的操作
    create_vector_store()