import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
load_dotenv()

# RAG Complete Steps

# 1. Select a chat model
# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# response = model.invoke("Write me a ballad about LangChain")

# print(response.get("content"))

# 2. Select an embeddings model
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
# print(embeddings_model.embed_query("Hello world!"))


# 3. Select a vector store
vector_store = Chroma(
    collection_name="text-collection",
    embedding_function=embeddings_model,
    persist_directory="./chroma_langchain_docs_db"
)

# 4. Indexing documents
"""
a. Load documents
b. Split documents into chunks
c. Create embeddings for each chunk
d. Store the embeddings in the vector store
"""




