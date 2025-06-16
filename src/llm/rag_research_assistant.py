import os
import glob
import faiss
import pickle
from typing import List
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings  # Replace with DeepSeek when available
from langchain.document_loaders import TextLoader, PyPDFLoader


CORPUS_DIR = "data/regulations_corpus/"
INDEX_PATH = "data/processed/aml_index"


def load_documents(folder : str) -> List[Document]:
    docs = []

    for file in glob.glob(os.path.join(folder, "*")):
        if file.endswith(".txt"):
            loader = TextLoader(file)
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            continue

        docs.extend(loader.load())

    return docs

def build_index():
    print("Loading documents...")
    raw_docs = load_documents(CORPUS_DIR)

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    print("Embedding and indexing...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Switch to DeepSeek later
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    print("Saving index...")
    vectorstore.save_local(INDEX_PATH)

def query_aml_policy(user_query: str, k: int = 3):
    print("Loading FAISS index...")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)

    print(f"Searching top {k} chunks for query: {user_query}")
    results = vectorstore.similarity_search(user_query, k=k)

    context = "\n---\n".join([doc.page_content for doc in results])

    prompt = f"""
You are an AML regulatory assistant. Based on the following AML policy content, answer the user's question.

Context:
{context}

Question: {user_query}

Answer concisely with references to guidance if possible.
"""

    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1")
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are an expert in financial crime regulations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content



if __name__ == "__main__":
    if not os.path.exists(INDEX_PATH):
        print("Index not found, building...")
        build_index()
    else:
        print("Index already exists, skipping build.")

    user_question = "What are the key red flags in wire transfers to shell companies?"
    response = query_aml_policy(user_question)
    print("\n📘 RAG Response:\n", response)
