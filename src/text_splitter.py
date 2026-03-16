from langchain.text_splitter import RecursiveCharacterTextSplitter


def split_documents(documents):
    """
    Split documents into smaller chunks for better retrieval.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    return chunks
