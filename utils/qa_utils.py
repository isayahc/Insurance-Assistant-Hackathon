from typing import Any
# from your_module import CohereEmbeddings, CharacterTextSplitter, PyPDFLoader, Chroma  # Replace 'your_module' with the actual module name

from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import TextSplitter

from langchain.embeddings import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.vectorstores import VectorStore

from langchain.vectorstores import Chroma

from langchain.document_loaders import PyPDFLoader


def create_embeddings_from_pdf(
        pdf_doc: str, 
        embeddings:Embeddings,
        db:VectorStore,
        chunk_size=350,
        chunk_overlap=0,
        ) -> VectorStore:
    """
    This function loads a PDF document using PyPDFLoader, splits its text, and creates embeddings
    From embedding model and store is into a VectorStore object

    :param pdf_doc: string location of pdf
    :param embedding: embedding model to embed text
    :param chunk_size: Optional chunk size for tokens. (Default 350).
    :param chunk_overlap: Optional. The overlap size between chunks. (Default  0).
    :return: The db object containing the document embeddings.
    """


    text_splitter:TextSplitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    loader = PyPDFLoader(pdf_doc.name)
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    db:VectorStore = Chroma.from_documents(texts, embeddings)
    
    return db
