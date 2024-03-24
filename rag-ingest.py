# keep data in data folder
# run this program before running chainlit program
# ensure that ollama server is running

import os
import warnings
from chromadb.config import Settings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
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
    #     chunk_size=500, chunk_overlap=40)
    text_splitter = NLTKTextSplitter()
    chunked_documents = text_splitter.split_documents(loaded_documents)
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
