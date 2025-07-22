# docs/index_docs.py
import os
import glob
import traceback
import argparse
import json
from tqdm import tqdm
import torch

# LangChain 相关库导入
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- 默认配置 ---
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
ADD_BATCH_SIZE = 100
# --- 嵌入模型的批处理大小，用于GPU计算。可根据显存大小调整以提升性能。---
EMBED_BATCH_SIZE = 1024

# --- JSON 文件加载器 (无变动) ---
def load_cail_scm_from_json(files_to_process: list) -> list:
    """
    从给定的文件列表加载 CAIL-SCM 格式的 JSON 文件,
    并将每个案件的 A, B, C 文书解析为独立的 LangChain Document 对象。

    :param files_to_process: 需要处理的JSON文件路径列表。
    :return: 一个包含所有解析出的 Document 对象的列表。
    """
    documents = []
    doc_id_counter = 0
    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"  - 正在处理新文件: {file_name}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc=f"  解析 {file_name}", unit="行")):
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    for key in ['A', 'B', 'C']:
                        if key in data and data[key]:
                            doc = Document(
                                page_content=data[key],
                                metadata={
                                    "source": file_name,
                                    "doc_id": f"case_{doc_id_counter}"
                                }
                            )
                            documents.append(doc)
                            doc_id_counter += 1
        except json.JSONDecodeError as e:
            print(f"❌ 错误: 解析 JSON 文件 '{file_path}' 的第 {i+1} 行时出错: {e}")
        except Exception as e:
            print(f"❌ 错误: 读取或处理文件 '{file_path}' 时出错: {e}")
            
    return documents


def create_vector_store(source_dir: str, db_dir: str, model_path: str, doc_type: str):
    """
    通用函数：根据文档类型加载文档, 分割, 创建向量存储。
    现在支持增量更新。
    """
    if not os.path.isdir(source_dir):
        print(f"❌ 错误：源目录 '{source_dir}' 不存在。")
        return
    if not os.path.isdir(model_path):
        print(f"❌ 错误：指定的本地模型路径不存在: '{model_path}'")
        return

    log_file_path = os.path.join(db_dir, 'processed_files.log')
    processed_files = set()
    if os.path.exists(log_file_path):
        with open(log_file_path, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f)
    print(f"📖 已处理文件日志找到 {len(processed_files)} 条记录。")

    # 1. 加载文档 (只加载新文件)
    print(f"\n📚 正在从 '{source_dir}' 检查并加载新的 {doc_type} 文档...")
    
    files_to_process = []
    if doc_type == 'legal':
        all_files = glob.glob(os.path.join(source_dir, '*.docx'))
        files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]
    elif doc_type == 'case':
        all_files = glob.glob(os.path.join(source_dir, '**', '*.json'), recursive=True)
        files_to_process = [f for f in all_files if os.path.basename(f) not in processed_files]

    if not files_to_process:
        print("✅ 未发现需要处理的新文件。数据库已是最新。")
        return

    print(f"📂 发现 {len(files_to_process)} 个新文件需要处理: {[os.path.basename(f) for f in files_to_process]}")
    
    documents = []
    try:
        if doc_type == 'legal':
            for file_path in files_to_process:
                 loader = Docx2txtLoader(file_path)
                 documents.extend(loader.load())
        elif doc_type == 'case':
            documents = load_cail_scm_from_json(files_to_process)
    except Exception as e:
        print(f"❌ 错误：加载新文档时出错: {e}"); traceback.print_exc(); return
    
    if not documents:
        print(f"⚠️ 警告：未能从新文件中加载任何文档内容。")
        return
        
    print(f"✅ 成功从新文件中加载了 {len(documents)} 个文档对象。")

    # 2. 分割文档
    print("\n✂️ 正在将新文档分割成块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", "、", " ", ""],
        keep_separator=False
    )
    docs_splitted = text_splitter.split_documents(documents)
    print(f"✅ 已将新文档分割成 {len(docs_splitted)} 个文本块。")

    # 3. 初始化嵌入模型
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🧠 自动检测到可用设备: {device.upper()}")
    # --- 修正：移除与LangChain内部调用冲突的 'show_progress_bar' 参数 ---
    embeddings = SentenceTransformerEmbeddings(
        model_name=model_path, 
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': EMBED_BATCH_SIZE}
    )

    # 4. 更新向量存储
    abs_db_dir = os.path.abspath(db_dir)
    print(f"\n💾 准备向向量存储库添加新数据: {abs_db_dir}")
    os.makedirs(abs_db_dir, exist_ok=True)
    try:
        vector_store = Chroma(persist_directory=abs_db_dir, embedding_function=embeddings)
        print(f"⏳ 开始分批添加 {len(docs_splitted)} 个新文本块 (批大小: {ADD_BATCH_SIZE})...")
        for i in tqdm(range(0, len(docs_splitted), ADD_BATCH_SIZE), desc="嵌入并存储", unit="批"):
            batch = docs_splitted[i:i + ADD_BATCH_SIZE]
            vector_store.add_documents(documents=batch)
        print("\n⏳ 正在持久化数据库...")
        vector_store.persist()
        
        print(f"✍️ 正在更新处理日志: {log_file_path}")
        with open(log_file_path, 'a', encoding='utf-8') as f:
            for file_path in files_to_process:
                f.write(os.path.basename(file_path) + '\n')

        print(f"🎉 向量存储更新成功！")
    except Exception as e:
        print(f"❌ 错误：在嵌入或存储到 Chroma 时出错: {e}"); traceback.print_exc()
    finally:
        vector_store = None; embeddings = None; import gc; gc.collect()

# --- 主程序入口 (无变动) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="为法律或案例文档创建向量数据库。")
    parser.add_argument('--type', type=str, choices=['legal', 'case'], required=True, help="要索引的文档类型: 'legal' (法条) 或 'case' (案例)。")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    if args.type == 'legal':
        SOURCE_DIRECTORY = os.path.join(script_dir, "legal")
        PERSIST_DIRECTORY = os.path.join(script_dir, "legal_db")
        print("--- 选择模式: 法条 (legal) ---")
    else: 
        SOURCE_DIRECTORY = os.path.join(script_dir, "case", "CAIL2019-SCM")
        PERSIST_DIRECTORY = os.path.join(script_dir, "case_db")
        print("--- 选择模式: 案例 (case) ---")
    
    base_dir = os.path.dirname(project_root)
    EMBEDDING_MODEL_PATH = os.path.join(base_dir, "models", "m3e-base")

    print("-" * 60)
    print(f"准备为 '{args.type}' 类型文档创建/更新向量数据库")
    print(f"  - 源文件目录: {os.path.abspath(SOURCE_DIRECTORY)}")
    print(f"  - 目标数据库: {os.path.abspath(PERSIST_DIRECTORY)}")
    print(f"  - 嵌入模型:   {os.path.abspath(EMBEDDING_MODEL_PATH)}")
    print("-" * 60)

    create_vector_store(
        source_dir=SOURCE_DIRECTORY,
        db_dir=PERSIST_DIRECTORY,
        model_path=EMBEDDING_MODEL_PATH,
        doc_type=args.type
    )

    print("-" * 60)
    print("脚本执行完毕。")
    print("-" * 60)