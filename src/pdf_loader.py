import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(data_path="data"):
    """
    Load all PDF files from the data folder and return LangChain documents.
    """

    documents = []

    for file in os.listdir(data_path):

        if file.lower().endswith(".pdf"):

            file_path = os.path.join(data_path, file)

            loader = PyPDFLoader(file_path)

            pdf_docs = loader.load()

            documents.extend(pdf_docs)

    return documents
