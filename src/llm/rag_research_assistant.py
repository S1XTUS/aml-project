import os
import sys
import glob
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings  # requires OPENAI_API_KEY

from src.llm.llm_client import get_llm_client
from src.utils.config_loader import config_path, load_config

CORPUS_DIR = config_path("regulations_corpus")
INDEX_PATH = config_path("rag_index")
EMBEDDING_MODEL = "text-embedding-ada-002"


def load_documents(folder: str) -> List[Document]:
    docs = []

    for file in glob.glob(os.path.join(folder, "*")):
        if file.endswith(".txt"):
            loader = TextLoader(file, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file)
        else:
            continue

        docs.extend(loader.load())

    return docs

def build_index():
    print("Loading documents...")
    raw_docs = load_documents(CORPUS_DIR)
    if not raw_docs:
        raise FileNotFoundError(
            f"No .txt or .pdf documents found in {CORPUS_DIR}. "
            "Add AML regulations / guidance (e.g. FATF recommendations, FinCEN advisories) first."
        )

    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    print("Embedding and indexing...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)

    print("Saving index...")
    vectorstore.save_local(INDEX_PATH)

def query_aml_policy(user_query: str, k: int = 3):
    print("Loading FAISS index...")
    # The index is created locally by build_index(), so loading its pickle is trusted
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings=OpenAIEmbeddings(model=EMBEDDING_MODEL),
                                   allow_dangerous_deserialization=True)

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

    response = get_llm_client().chat.completions.create(
        model=load_config()["llm"]["model"],
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
