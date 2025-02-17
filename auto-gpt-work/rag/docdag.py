import os
from dotenv import load_dotenv
#from docx import Document
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings  # 添加这一行

# 加载环境变量（如果需要）
load_dotenv()

# 1. 处理PDF文档

# 1. 处理文档（支持 PDF 和 DOCX）
def process_document(file_path):
    # 根据文件类型加载不同的Loader
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("不支持的文件格式。只支持 PDF 和 DOCX 文件。")

    pages = loader.load()

    # 分割文本
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    return texts

# 2. 向量存储配置
def setup_vector_store(docs):
    # 初始化嵌入模型
    #embeddings = HuggingFaceEmbeddings(
    #    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    #)
    embeddings = OpenAIEmbeddings(
       model="text-embedding-3-large",
       dimensions=1024  # 可选：降维到 1024（默认 3072）
)

 
    
    # 连接Milvus
    # 确保从Milvus文档创建时，正确配置`auto_id`，通常在创建集合时配置
    vector_db = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        connection_args={
            "uri": "http://localhost:19530",  # Milvus服务器地址
            "token": "",  # 如果有认证需要填写
            "auto_id": True,  # 将auto_id配置移到connection_args
        },
        collection_name="rag_demo",
        drop_old=True  # 每次重建集合
    )
    return vector_db

# 3. 自定义排序：根据相似度等进行重排序
def custom_sort_documents(docs):
    """
    对文档按相似度进行重新排序。
    可以根据具体需求调整排序逻辑，当前例子按文档的相似度得分排序。
    """
    # 这里使用余弦相似度的得分来排序（假设已经有每个文档的相似度得分）。
    sorted_docs = sorted(docs, key=lambda doc: doc.metadata["source"], reverse=True)
    return sorted_docs
# 4. 构建RAG链
def create_rag_chain(vector_db, history):
    # 定义检索器
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
      # 获取前k个相关文档并进行重排序
    def get_relevant_documents_with_sorting(query):
        # 获取前k个相关文档
        docs = retriever.get_relevant_documents(query)
        # 对文档进行排序
        sorted_docs = custom_sort_documents(docs)
        return sorted_docs[:3]  # 返回前3个文档作为最终结果
    
    # 定义提示模板，增加历史会话的内容
    template = """以下是你和我之间的对话记录：
    {history}
    
    基于以下上下文信息回答问题：
    {context}
    
    问题：{question}
    请用中文给出详细回答："""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # 初始化LLM（这里使用本地模型示例，需根据实际情况修改）
    #llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # 替换为实际使用的模型
    

    llm = ChatOpenAI(model="deepseek-chat",
                   api_key="sk-01f770c916d1424c84ed60dd7fd78908",
                   base_url="https://api.deepseek.com"
                   )

    # 构建处理链
    rag_chain = (
        {"context": get_relevant_documents_with_sorting, "question": RunnablePassthrough(), "history": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
# 3. 验证生成结果
def validate_answer(answer, retriever):
    """
    验证答案的有效性。
    例如：检查生成的答案是否相关，是否与检索的文档一致。
    """
    if not answer:
        return False  # 如果没有答案，直接返回False

    # 检查答案是否包含了检索到的上下文内容
    #relevant_docs = retriever.get_relevant_documents(answer)
    relevant_docs = retriever.invoke(answer)
    for doc in relevant_docs:
        if doc.page_content.lower() in answer.lower():
            return True
    
    # 如果检索到的文档中没有相关内容在答案中，返回False
    return False


# 主程序
if __name__ == "__main__":
    # 处理PDF文档
    pdf_path = "F:\AI\/ragtest\hrg.pdf"  # 替换为你的PDF文件路径
    doc_path ="F:\AI\/ragtest\lbg.docx"
    documents = process_document(pdf_path)
    documents = process_document(doc_path)
    # 设置向量数据库
    vector_store = setup_vector_store(documents)
    
    # 历史会话初始化为空
    conversation_history = []
    
    # 创建RAG链
    chain = create_rag_chain(vector_store, conversation_history)
    
    # 开始对话
    print("RAG系统已就绪，输入'exit'退出")
    while True:
        query = input("\n请输入问题：")
        if query.lower() == "exit":
            break
        
        response = chain.invoke(query)
        print("\n回答：")
        print(response)

  # 验证回答的有效性
      #  retriever = vector_store.as_retriever(search_kwargs={"k": 5})
       # is_valid = validate_answer(response, retriever)
       # if is_valid:
       #     print("\n回答验证通过：答案与检索的文档一致。")
       # else:
       #     print("\n回答验证失败：答案与检索的文档不一致。")

         # 更新历史会话
        conversation_history.append(f"问题：{query}")
        conversation_history.append(f"回答：{response}")
        
