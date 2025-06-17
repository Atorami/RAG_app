# Imnport lib for loading documents
from langchain.document_loaders import DirectoryLoader
# Import lib for text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter


#### 1.Document Loading and Processing ####

# Directory containing the documents to be loaded

DATA_PATH = "data"


# Load documents from directory
def load_documents():

    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf")
    documents = loader.load()
    return documents

# Splitting documents into smaller chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    return chunks

#### 2.Embedding Creation ####