import pymupdf
import chromadb
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com")

def appText(messages,content):
    text = {"role": "user", 'content': content}
    messages.append(text)
    return messages

def appPrintText(messages,aiContent):
    printText = {"role": "assistant", 'content': aiContent}
    messages.append(printText)
    return messages
def clean_text(text):
    """
    清洗文本
    :param text:
    :return:text
    """
    end_marks = "?.!？。！…"
    lines = text.split("\n")
    paragraph = ""
    result = []
    for line in lines:
        line = line.rstrip()
        if not line:
            continue
        paragraph += line
        if line[-1] in end_marks:
            result.append(paragraph)
            paragraph = ""
    if paragraph:
        result.append(paragraph)
    return "\n".join(result)

def chunk_text(text,size=300,overlap=50):
    chunks = []
    start = 0
    while start <len(text):
        end = start+size
        chunks.append(text[start:end])
        start=end-overlap

    return chunks

doc = pymupdf.open(r"C:\Users\31155\Downloads\839a0cfd378b445a8499be5eabe82a6d.pdf")
print("总页数:",doc.page_count)
fullTextList =[]
for i,page in enumerate(doc):
    text=page.get_text()
    text=clean_text(text)
    fullTextList.append(text)

fullText=""
for i in fullTextList:
    fullText+=i

chunks = chunk_text(fullText,size=300,overlap=50)
print("总块数：",len(chunks))



# 持久化客户端：会在项目目录建 chroma_db 文件夹，下次运行数据不会丢失
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_or_create_collection(name="gov_docs")

# 入库：documents 直接传文版，chromadb 自动向量化
collection.add(
    documents = chunks,
    ids = [f"chunk_{i}" for i in range(len(chunks))],
    metadatas = [{"source":"培训平台操作手册"} for _ in chunks]
)

# 检索：用一个问题去库里找最相关的2块
results = collection.query(
    query_texts = ["线下的学时怎么申报？"],
    n_results = 2
)

# 打印召回的原文和距离（距离越小 = 予以越接近）
for doc,dist in zip(results["documents"][0],results["distances"][0]):
    print("----距离：",round(dist,3),"-----")
    print(doc)

context = "\n\n".join(results["documents"][0])

prompt = f"""请根据下面的参考资料回答问题，资料中没有的内容就回答"根据现有资料无法回答"，不要编造。
参考资料：
{context}

问题：线下的学时怎么申报？
"""

message = [
    {"role": "system", "content": "你是政务资料问答助手，只能依据提供的资料回答。"},
    {"role": "user", "content": prompt}
]


response = client.chat.completions.create(
    model = "deepseek-chat",
    messages = message,
    stream=True
)
aiContent =""
for i in response:
    if i.choices[0].delta.content is not None:
        aiContent += i.choices[0].delta.content
        print(i.choices[0].delta.content,end="")
print()
message = appPrintText(message, aiContent)