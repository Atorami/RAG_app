## RAG Question-Answering System for Museum Documentation
This project implements a Retrieval-Augmented Generation (RAG) based question-answering system designed to work with museum-related textual documentation. It allows users to ask questions and receive contextually accurate answers grounded in source documents.

### Features
- Load and process museum documents from a folder

- Split documents into overlapping chunks (500–1000 tokens with 50–100 token overlap)

- Generate embeddings and store them in a vector database (FAISS / ChromaDB)

- Integrate with a Large Language Model (e.g., OpenAI GPT) for answer generation

- Retrieve relevant document chunks before generation

- Expose a REST API for question answering

- (Optional) Implement caching, logging, and response time metrics

### Tech Stack
- Python 3.10+

- FastAPI

- LangChain

- ChromaDB

- OpenAI Embeddings

- OpenAI API / Hugging Face Transformers

### Architecture Overview
- Indexing

- Load and chunk documents

- Generate and store embeddings in a vector database

- Question Answering:

- Receive user question

- Retrieve top relevant chunks using vector similarity search

- Use LLM to generate an answer based on retrieved context