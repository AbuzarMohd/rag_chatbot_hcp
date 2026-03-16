import os
from langchain_community.document_loaders import PyPDFLoader


def load_pdfs(data_path="data"):
    """
    Load all PDF files from the data folder.
    Skip corrupted PDFs.
    """

    documents = []

    for file in os.listdir(data_path):

        if file.lower().endswith(".pdf"):

            file_path = os.path.join(data_path, file)

            try:
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                documents.extend(pdf_docs)

            except Exception as e:
                print(f"Skipping {file} due to error: {e}")

    return documents
