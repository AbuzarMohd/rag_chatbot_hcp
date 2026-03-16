from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings


def create_vector_store(chunks, api_key):
    """
    Create a vector database from document chunks.
    """

    embeddings = OpenAIEmbeddings(api_key=api_key)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectorstore
