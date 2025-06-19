# docs/index_legal_docs.py
import os
import glob
import traceback
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# --- 默认配置 (不变) ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_BATCH_SIZE = 100

def create_vector_store(source_dir: str, db_dir: str, model_path: str):
    """
    通用函数：加载指定目录的 DOCX, 分割, 从本地加载嵌入模型, 创建向量存储。
    (此函数内部逻辑完全不需要修改)
    """

    # 0. 检查路径
    if not os.path.isdir(source_dir):
        print(f"❌ 错误：源目录 '{source_dir}' 不存在。")
        return
    if not os.path.isdir(model_path):
        print(f"❌ 错误：指定的本地模型路径不存在: '{model_path}'")
        return

    # 1. 加载文档
    print(f"\n📚 正在从 '{source_dir}' 加载 DOCX 文档...")
    loader = DirectoryLoader(
        source_dir, glob="*.docx", loader_cls=Docx2txtLoader,
        show_progress=True, use_multithreading=False, recursive=False, silent_errors=True
    )
    try:
        documents = loader.load()
        if not documents:
            print(f"⚠️ 警告：在目录 '{source_dir}' 中未找到或未能加载任何 .docx 文件。")
            return
    except ImportError as e:
        if 'docx2txt' in str(e).lower(): print(f"❌ 错误：缺少 'docx2txt' 库。请运行 'pip install docx2txt'。")
        else: print(f"❌ 导入错误: {e}")
        return
    except Exception as e:
        print(f"❌ 错误：加载 DOCX 文档时出错: {e}"); traceback.print_exc(); return
    print(f"✅ 成功加载了 {len(documents)} 个文档对象。")

    # 2. 分割文档
    print("\n✂️ 正在将文档分割成块...")
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
            keep_separator=False
        )
        docs_splitted = text_splitter.split_documents(documents)
    except Exception as e:
        print(f"❌ 错误：分割文档时出错: {e}"); traceback.print_exc(); return
    if not docs_splitted:
        print("❌ 错误：未能成功分割任何文档块。"); return
    print(f"✅ 已将文档分割成 {len(docs_splitted)} 个文本块。")

    # 3. 初始化嵌入模型
    print(f"\n🧠 正在从本地路径加载嵌入模型: {model_path}")
    try:
        embeddings = SentenceTransformerEmbeddings(
            model_name=model_path,
            model_kwargs={'device': 'cpu'}
        )
        _ = embeddings.embed_query("测试嵌入模型加载") # 触发加载
    except Exception as e:
        print(f"❌ 错误：从 '{model_path}' 初始化嵌入模型时出错: {e}")
        traceback.print_exc()
        return
    print("✅ 嵌入模型初始化成功。")

    # 4. 创建/更新向量存储
    abs_db_dir = os.path.abspath(db_dir)
    print(f"\n💾 准备创建/更新向量存储于: {abs_db_dir}")
    os.makedirs(abs_db_dir, exist_ok=True)
    try:
        vector_store = Chroma(persist_directory=abs_db_dir, embedding_function=embeddings)
        print(f"⏳ 开始分批添加 {len(docs_splitted)} 个文本块 (批大小: {ADD_BATCH_SIZE})...")
        for i in tqdm(range(0, len(docs_splitted), ADD_BATCH_SIZE), desc="嵌入并存储", unit="批"):
            batch = docs_splitted[i:i + ADD_BATCH_SIZE]
            vector_store.add_documents(documents=batch)
        print("\n⏳ 正在持久化数据库...")
        vector_store.persist()
        print(f"🎉 向量存储创建/更新成功！数据库位于 '{abs_db_dir}'。")
    except Exception as e:
        print(f"❌ 错误：在嵌入或存储到 Chroma 时出错: {e}"); traceback.print_exc()
    finally:
        # 清理内存
        vector_store = None
        embeddings = None
        import gc
        gc.collect()


if __name__ == "__main__":
    # --- 命令行参数设置 (不变) ---
    parser = argparse.ArgumentParser(description="为法律或案例文档创建向量数据库。")
    parser.add_argument(
        '--type',
        type=str,
        choices=['legal', 'case'],
        required=True,
        help="要索引的文档类型: 'legal' (法条) 或 'case' (案例)。"
    )
    args = parser.parse_args()

    # --- 路径动态计算 (这里是唯一的修改) ---
    # 脚本位于 multi_agent/docs/
    script_dir = os.path.dirname(os.path.abspath(__file__)) # .../docs
    project_root = os.path.dirname(script_dir)              # .../multi_agent

    # 根据参数设置源目录和数据库目录
    if args.type == 'legal':
        # 源目录是当前目录下的 'legal' 子目录
        SOURCE_DIRECTORY = os.path.join(script_dir, "legal")
        # 目标目录是当前目录下的 'legal_db' 子目录
        PERSIST_DIRECTORY = os.path.join(script_dir, "legal_db")
        print("--- 选择模式: 法条 (legal) ---")
    else: # args.type == 'case'
        # 源目录是当前目录下的 'case' 子目录
        SOURCE_DIRECTORY = os.path.join(script_dir, "case")
        # 目标目录是当前目录下的 'case_db' 子目录
        PERSIST_DIRECTORY = os.path.join(script_dir, "case_db")
        print("--- 选择模式: 案例 (case) ---")
    
    # 模型路径需要从项目根目录计算
    EMBEDDING_MODEL_PATH = os.path.join(project_root, "models", "m3e-base")
    # --- 路径计算修改结束 ---

    # --- 执行 (不变) ---
    print("-" * 60)
    print(f"准备为 '{args.type}' 类型文档创建向量数据库")
    print(f"  - 源文件目录: {os.path.abspath(SOURCE_DIRECTORY)}")
    print(f"  - 目标数据库: {os.path.abspath(PERSIST_DIRECTORY)}")
    print(f"  - 嵌入模型:   {EMBEDDING_MODEL_PATH}")
    print("-" * 60)

    create_vector_store(
        source_dir=SOURCE_DIRECTORY,
        db_dir=PERSIST_DIRECTORY,
        model_path=EMBEDDING_MODEL_PATH
    )

    print("-" * 60)
    print("脚本执行完毕。")
    print("-" * 60)