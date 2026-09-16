from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# 1. 示例文本和嵌入模型
texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]
#把普通字符串列表 texts 转成 LangChain 需要的 Document 对象列表
docs = [Document(page_content=t) for t in texts]
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

#创建向量并存储本地
vectorstore = FAISS.from_documents(docs,embeddings)

#存储本地
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)

print(f"FAISS index has been saved to {local_faiss_path}")

# 加载索引并执行查询
# 加载的时候指定相同的模型，并允许反序列化
loaded_vectorstore = FAISS.load_local(
    local_faiss_path,
    embeddings,
    allow_dangerous_deserialization=True
)

#相似性搜索
query = "FAISS是做什么的？"
result = loaded_vectorstore.similarity_search(query,k =1)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in result:
    print(f"- {doc.page_content}")