import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(data_path="data"):
    """
    Load all PDF files from the data folder.
    """

    documents = []

    for file in os.listdir(data_path):

        if file.endswith(".pdf"):

            file_path = os.path.join(data_path, file)

            loader = PyPDFLoader(file_path)

            docs = loader.load()

            documents.extend(docs)

    return documents
