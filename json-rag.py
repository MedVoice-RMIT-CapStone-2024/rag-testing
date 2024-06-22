from langchain_community.llms import Ollama 
from langchain_community.document_loaders import JSONLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class RAGChatbot:
    def __init__(self):        
        # Initialize the Local Model
        self.llm = Ollama(model="llama3", temperature=0)
        self.vectorstore = None
        self.rag_chain = None
    
    def index_json_folder(self, file_path):
        loader = JSONLoader(file_path, jq_schema=".prizes[]", text_content=False)
        docs = loader.load()
        self._index_documents(docs)
        return {"message": "JSON files indexed successfully"}

    def _index_documents(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        all_splits = text_splitter.split_documents(docs)
        
        # Initialize the Embeddings Model with GPU support if available
        embedding = OllamaEmbeddings(model="nomic-embed-text")
        
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=embedding,
            )
        else:
            self.vectorstore.add_documents(all_splits)
        
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )
        
        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt 
            | self.llm 
            | StrOutputParser()
        )
    
    def query_model(self, question: str):
        if self.rag_chain is None:
            return {"error": "No documents have been indexed yet."}
        
        answer_0 = self.llm.invoke(question)
        answer = self.rag_chain.invoke(question)
        return {"question": question, "answer before RAG": answer_0}, {"question": question, "answer after RAG": answer}

    def continuous_conversation(self):
        while True:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            response = self.query_model(question)
            print(f"Answer before RAG: {response[0]['answer before RAG']}")
            print(f"Answer after RAG: {response[1]['answer after RAG']}")

# Example usage
chatbot = RAGChatbot()

# Index JSON files from a folder
file_path = "prize.json"
print(chatbot.index_json_folder(file_path))

# Start continuous conversation
chatbot.continuous_conversation()
