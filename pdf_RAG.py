from langchain.chains import LLMChain
from langchain_community.llms import Replicate
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import os
import json

def initialize_llm() -> Replicate:
    # Initialize the Replicate instance
    llm = Replicate(
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        model="meta/meta-llama-3-70b-instruct",
        model_kwargs= {
            "top_k": 0,
            "top_p": 0.9,
            "max_tokens": 512,
            "min_tokens": 0,
            "temperature": 0.6,
            "length_penalty": 1,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "presence_penalty": 1.15,
            "log_performance_metrics": False
        },
    )
    return llm

def index_pdf(file_path):
    # Initialize the Local Model
    global llm
    llm = initialize_llm()

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
    answer = rag_chain.invoke(question)
    return {"question": question, "answer after RAG": answer}

# Example usage
file_path = "./update-28-covid-19-what-we-know.pdf"
print(json.dumps(index_pdf(file_path), indent=4))

# Interactive RAG System
conversation_state = {}
while True:
    question = input("Please enter your question or type 'exit' to end the conversation: ")
    if question.lower() == 'exit':
        break
    elif question.lower() == 'new':
        conversation_state = {}
        print("Starting a new conversation...")
        continue
    elif question in conversation_state:
        print(json.dumps(conversation_state[question], indent=4))
    else:
        response = query_model(question)
        conversation_state[question] = response
        print(json.dumps(response, indent=4))
