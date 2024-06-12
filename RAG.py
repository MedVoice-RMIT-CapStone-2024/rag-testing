from langchain_community.llms import Ollama 
from langchain_community.document_loaders import WebBaseLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub 
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize the Local Model
llm = Ollama(model="llama3", temperature=0)

# Indexing: Load
loader = WebBaseLoader(
    web_path="https://isqua.org/blog/covid-19/covid-19-blogs.html",
)
docs = loader.load()
# print(docs[0].page_content)
# print(len(docs[0].page_content))

# Indexing: Split
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)
# print(len(all_splits))
# print(len(all_splits[1].page_content))

# Indexing: Store
embedding = OllamaEmbeddings(
    model="nomic-embed-text",
)
vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embedding,
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6},
)

# Retrieval and Generation: Generate
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)

# Queries
print("Before applying RAG technique:\n")
print("How to handle covid 19?\n")
print(llm.invoke("How to handle covid 19?"))
print("\n\n")

print("After applying RAG technique:\n")
print("How to handle covid 19?\n")
print(rag_chain.invoke("How to handle covid 19?"))

