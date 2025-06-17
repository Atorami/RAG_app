#### Part 1: Data Preparation and Vectorization ####

# Imnport lib for loading documents and pdf loader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
# Import lib for text splitting
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
     # Checking if the chroma path exists, if not create it
    if not os.path.exists(CHROMA_PATH):
        os.makedirs(CHROMA_PATH)

    # Initialize ChromaDB client with DuckDB and Parquet storage
    chroma_client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet", 
        persist_directory=CHROMA_PATH
    ))

    # Create or get the collection 
    collection = chroma_client.get_or_create_collection(name="documents")
    
    # Adding documents to the collection
    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(texts))]
    )

    # Saving to disk
    chroma_client.persist()
    return collection

collection = store_embeddings(texts, embeddings, metadatas)



#### Part 2: LLM Integration ####

#### 1.LLM Setup ####

from transformers import AutoModelForCausalLM, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForCausalLM.from_pretrained("google/flan-t5-base")