from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path):
    """
    Load a PDF file and convert it into LangChain document objects.
    """

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    return documents
