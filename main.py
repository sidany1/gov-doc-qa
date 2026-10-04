import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 保留镜像，以后下新模型用
os.environ["HF_HUB_OFFLINE"] = "1"                    # 强制离线，只用本地缓存，不联网检查
from dotenv import load_dotenv
load_dotenv()

import chromadb
from openai import OpenAI
import pymupdf

PDF_DIR = r"C:\Users\31155\Downloads\pdfs"  # 存放多个PDF的文件夹

# 按页清洗：去掉空行，按句子（？！。）拼接
def clean_page_text(page):
    lines = page.get_text().rstrip().split("\n")
    sentences = []
    text2 = ""
    for i in lines:
        i = i.rstrip()
        if not i:
            continue
        text2 += i
        if i[-1] in "？！。":
            sentences.append(text2)
            text2 = ""
    if text2:
        sentences.append(text2)
    return "\n".join(sentences)

# 遍历文件夹中的所有PDF，按页切块，记录来源文件和页码
chunks = []
sources = []
page_nums = []
for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue
    pdf = pymupdf.open(os.path.join(PDF_DIR, filename))
    for page_num, page in enumerate(pdf, start=1):  # 页码从1开始
        cleanText = clean_page_text(page)
        if not cleanText:
            continue
        start = 0
        end = 300
        while start < len(cleanText):
            chunks.append(cleanText[start:end])
            sources.append(filename)
            page_nums.append(page_num)
            start = end - 50
            end = start + 300
    pdf.close()

db = chromadb.PersistentClient(path="./chroma_db")
from chromadb.utils import embedding_functions

bge_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="BAAI/bge-small-zh-v1.5"
)
table = db.get_or_create_collection("text_index_bge2", embedding_function=bge_ef)

# # 先清空旧数据，避免ID重复报错
# if table.count() > 0:
#     table.delete(ids=table.get()["ids"])

table.add(
    documents=chunks,
    ids=[f"ids_{i}" for i in range(len(chunks))],
    metadatas=[{"source": s, "page": p} for s, p in zip(sources, page_nums)]
)

prod = input("请输入要查询的文本：")
result = table.query(query_texts=[prod],n_results=2)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")
parts = []
pages_used = []
for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
    parts.append(f"【{meta['source']} 第{meta['page']}页】{doc}")
    pages_used.append(meta["page"])
context = "\n".join(parts)
resp = f"""问题：{prod}
参考资料：{context}
参考资料没有的请回答我不知道"""

# print(chat.choices[0].message.content)
# 收集出处（文件名 + 页码），多文档必须两个一起
refs = []
for meta in result["metadatas"][0]:
    refs.append((meta["source"], meta["page"]))

# set 对元组也能自动去重（同一文件同一页只留一次），sorted 排序
unique_refs = sorted(set(refs))
ref_str = "、".join(f"《{s}》第{p}页" for s, p in unique_refs)
closest = result["distances"][0][0]   # 最近一条的距离
if closest > 0.6:                     # 阈值要自己试，bge 下越小越相关
    print("未在知识库中找到相关资料。")
else:
    # 取消注释、调模型
    chat = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是政务资料问答助手，只能依据带出处的资料回答，资料没有就说我不知道。"},
                  {"role": "user", "content": resp}]
    )
    print(chat.choices[0].message.content)
    print(f"（资料来源：{ref_str}）")
