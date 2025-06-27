#### Part 1: Data Preparation and Vectorization ####

from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



#### 1.Document Loading and Processing ####

# Directory containing the documents to be loaded

DATA_PATH = "data"


# Load documents from directory
def load_documents():

    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

# Splitting documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f"Loaded {len(documents)} documents")
    print(f"Split into {len(chunks)} chunks")

    return chunks

#### 2.Embedding Creation ####

import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "chroma_db"

# Get chunks from loaded documents
documents = load_documents()
chunks = split_documents(documents)

texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]


# embedding text to vector lib
def create_embeddings(texts):

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(texts, convert_to_tensor=False)
    return embeddings


embeddings = create_embeddings(texts)


def store_embeddings(texts, embeddings, metadatas):
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

    collection = chroma_client.get_or_create_collection(name="documents")

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

    return collection

collection = store_embeddings(texts, embeddings, metadatas)



#### Part 2: LLM Integration ####

#### 1.LLM Setup ####

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model =AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def search_docs(query, collection, embedding_model, top_k=5):

    query_embedding = embedding_model.encode([query], convert_to_tensor=False)[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    
    return results['documents'][0], results['metadatas'][0]


def build_prompt(query, docs):
    context = "\n\n".join(docs)
    promt = (
        f"Context: {context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    return promt

def generate_answer(prompt, model, tokenizer, max_length=300):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=max_length)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def rag_pipeline(query, collection, embedding_model, llm_model, tokenizer):
    docs, metas = search_docs(query, collection, embedding_model)
    prompt = build_prompt(query, docs)
    answer = generate_answer(prompt, llm_model, tokenizer)
    return answer, docs, metas
