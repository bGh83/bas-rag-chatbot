# Notes
# Create python virtual environment and activate it
# python -m venv venv
# Install following dependencies withint the venv
# pip install langchain langchain-community langchain-core
# pip install chromadb
# pip install chainlit
# pip install sentence-transformers nltk
# keep data in data folder
# run this program before running chainlit program
# python .\rag-ingest.py
# ensure that ollama server is running
# run chainlit program
# chainlit run .\rag-chainlit.py

import os
import warnings
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

warnings.simplefilter("ignore")

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db-txt")


def create_vector_database():

    pdf_loader = DirectoryLoader(
        "data/", glob="**/*.txt", show_progress=True, use_multithreading=True, loader_cls=TextLoader)
    loaded_documents = pdf_loader.load()

    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=512, chunk_overlap=0, length_function=len)
    # text_splitter = NLTKTextSplitter()
    text_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0)
    chunked_documents = text_splitter.split_documents(loaded_documents)

    # print("\nBEGIN=============================================================")
    # print(len(chunked_documents))
    # for doc in chunked_documents:
    #     print(doc.page_content)
    #     print("\n>>>>>")
    # print("\nEND=============================================================")

    ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_database = Chroma.from_documents(
        documents=chunked_documents,
        embedding=ollama_embeddings,
        collection_name="rag-chroma",
        persist_directory=DB_DIR,
        client_settings=Settings(
            anonymized_telemetry=False, is_persistent=True)
    )

    vector_database.persist()

    query = "what are the default rules"
    docs = vector_database.similarity_search(query)
    print(docs[0].page_content)


if __name__ == "__main__":
    create_vector_database()
