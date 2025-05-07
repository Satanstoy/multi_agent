# legal_docs/index_legal_docs.py
import os
import glob
import traceback
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- 配置 ---
SOURCE_DIRECTORY = "./"  # 假设 DOCX 文件在此脚本所在的目录下
PERSIST_DIRECTORY = "legal_db"  # 向量数据库目录，相对于脚本运行位置

# --- 修改：指定包含模型文件的本地目录路径 ---
script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在目录 (/.../legal_docs)
project_root = os.path.dirname(script_dir) # 获取上一级目录 (/.../multi_agent)
# 明确指向 multi_agent/models/m3e-base 目录
EMBEDDING_MODEL_LOCAL_PATH = os.path.join(project_root, "models", "m3e-base")
# --- 本地路径配置结束 ---

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_BATCH_SIZE = 100
# --- 配置结束 ---

# --- 依赖提醒 ---
# pip install docx2txt tqdm sentence-transformers
# pip install langchain-community langchain-text-splitters chromadb
# --- 依赖提醒结束 ---

def create_vector_store():
    """加载 DOCX, 分割, 从本地路径加载嵌入模型, 创建向量存储"""

    # 0. 检查源目录
    if not os.path.isdir(SOURCE_DIRECTORY):
        print(f"❌ 错误：源目录 '{SOURCE_DIRECTORY}' 不存在。")
        return

    # 1. 查找文件
    print(f"🔍 正在查找目录 '{os.path.abspath(SOURCE_DIRECTORY)}' 中的 .docx 文件...")
    try:
        docx_files = glob.glob(os.path.join(SOURCE_DIRECTORY, "*.docx"), recursive=False)
        if not docx_files:
            print(f"⚠️ 警告：在目录 '{os.path.abspath(SOURCE_DIRECTORY)}' 中未找到任何 .docx 文件。")
            return
        print(f"  找到 {len(docx_files)} 个 .docx 文件。")
    except Exception as e:
        print(f"❌ 错误：查找文件时出错: {e}")
        return

    # 2. 加载文档
    print(f"\n📚 开始加载 {len(docx_files)} 个 DOCX 文档 (使用 Docx2txtLoader)...")
    loader = DirectoryLoader(
        SOURCE_DIRECTORY, glob="*.docx", loader_cls=Docx2txtLoader,
        show_progress=True, use_multithreading=False, recursive=False, silent_errors=True
    )
    try:
        documents = loader.load()
    except ImportError as e:
        if 'docx2txt' in str(e).lower(): print(f"❌ 错误：缺少 'docx2txt' 库。请运行 'pip install docx2txt'。")
        else: print(f"❌ 导入错误: {e}")
        return
    except Exception as e: print(f"❌ 错误：加载 DOCX 文档时出错: {e}"); traceback.print_exc(); return
    if not documents: print(f"⚠️ 警告：未能成功加载任何文档内容。"); return
    print(f"\n✅ 成功加载了 {len(documents)} 个文档对象。")

    # 3. 分割文档
    print("\n✂️ 正在将加载的文档内容分割成块...")
    try:
        # 在第 71 行附近
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP, # <-- 改回 chunk_overlap
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
            keep_separator=False
        )
        docs_splitted = []
        print("  (进度条基于处理的文档对象)")
        for doc in tqdm(documents, desc="分割文档对象", unit="文档"):
            try: chunks = text_splitter.split_documents([doc]); docs_splitted.extend(chunks)
            except Exception as split_err: source_file = doc.metadata.get('source', '未知文件'); print(f"\n⚠️ 警告：分割 '{os.path.basename(source_file)}' 时出错: {split_err}")
    except Exception as e: print(f"❌ 错误：初始化分割器时出错: {e}"); traceback.print_exc(); return
    if not docs_splitted: print("❌ 错误：未能成功分割任何文档块。"); return
    print(f"✅ 已将文档分割成 {len(docs_splitted)} 个文本块。")

    # --- 4. 初始化嵌入模型 (从本地路径加载) ---
    print(f"\n🧠 正在从本地路径加载嵌入模型: {EMBEDDING_MODEL_LOCAL_PATH}")

    # 检查指定路径是否存在
    if not os.path.isdir(EMBEDDING_MODEL_LOCAL_PATH):
        print(f"❌ 错误：指定的本地模型路径不存在或不是一个目录: '{EMBEDDING_MODEL_LOCAL_PATH}'")
        print(f"   请确保该路径正确，并且其中包含所有必要的模型文件 (config.json, pytorch_model.bin 等)。")
        return

    try:
        # --- 修改：直接使用本地路径作为 model_name ---
        embeddings = SentenceTransformerEmbeddings(
            model_name=EMBEDDING_MODEL_LOCAL_PATH, # <-- 使用本地路径
            model_kwargs={'device': 'cpu'}
            # 当直接指定本地路径时，不再需要 cache_folder 参数
        )
        # --- 修改结束 ---
        print("  测试嵌入模型加载...")
        _ = embeddings.embed_query("测试") # 触发加载
    except Exception as e:
        print(f"❌ 错误：从本地路径 '{EMBEDDING_MODEL_LOCAL_PATH}' 初始化嵌入模型时出错。")
        print(f"   请确保该路径下包含完整且有效的模型文件，并且 sentence-transformers 库已安装。错误: {e}")
        traceback.print_exc()
        return
    print("✅ 嵌入模型初始化/加载成功。")

    # --- 5. 创建/更新向量存储 ---
    # (逻辑不变)
    abs_persist_dir = os.path.abspath(PERSIST_DIRECTORY)
    print(f"\n💾 准备创建/更新向量存储于: {abs_persist_dir}")
    os.makedirs(abs_persist_dir, exist_ok=True)
    try:
        print("  初始化 Chroma 数据库连接...")
        vector_store = Chroma(persist_directory=abs_persist_dir, embedding_function=embeddings)
        print(f"⏳ 开始分批添加/嵌入 {len(docs_splitted)} 个文本块到数据库 (批大小: {ADD_BATCH_SIZE})...")
        for i in tqdm(range(0, len(docs_splitted), ADD_BATCH_SIZE), desc="嵌入并存储块", unit="批"):
            batch = docs_splitted[i:i + ADD_BATCH_SIZE]
            try: vector_store.add_documents(documents=batch)
            except Exception as add_err: print(f"\n⚠️ 警告：添加批次 {i//ADD_BATCH_SIZE + 1} 到数据库时出错: {add_err}")
        print("\n⏳ 正在持久化数据库 (确保所有更改已写入)...")
        vector_store.persist()
        # 清理 vector_store 对象和嵌入模型，尝试释放内存
        print("  清理内存...")
        vector_store = None
        embeddings = None
        import gc
        gc.collect() # 尝试强制垃圾回收
        print(f"🎉 向量存储创建/更新并持久化成功！数据库位于 '{abs_persist_dir}'。")
    except Exception as e: print(f"❌ 错误：在嵌入或存储块到 Chroma 时出错: {e}"); traceback.print_exc()


if __name__ == "__main__":
    print("-" * 50)
    print("开始创建/更新向量数据库 (使用 Docx2txtLoader, 从本地加载嵌入模型)...") # 更新提示
    print("-" * 50)
    create_vector_store()
    print("-" * 50)
    print("脚本执行完毕。")
    print("-" * 50)