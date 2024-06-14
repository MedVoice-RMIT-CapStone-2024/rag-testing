from langchain_community.llms import Ollama 
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

def index_pdf(file_path):
    # Initialize the Local Model
    global llm
    llm = Ollama(model="llama3", temperature=0)

    # Indexing: Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Indexing: Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    
    # Indexing: Store
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=embedding,
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6},
    )
    
    # Store retriever globally for later use
    global rag_chain
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt 
        | llm 
        | StrOutputParser()
    )
    
    return {"message": "PDF indexed successfully"}

def query_model(question: str):
    if 'rag_chain' not in globals():
        return {"error": "No documents have been indexed yet."}
    answer_0 = llm.invoke(question)
    answer = rag_chain.invoke(question)
    return {"question": question, "answer before RAG": answer_0}, {"question": question, "answer after RAG": answer}

# Example usage
file_path = "./update-28-covid-19-what-we-know.pdf"
print(index_pdf(file_path))

question = "What is Covid 19?"
print(query_model(question))
