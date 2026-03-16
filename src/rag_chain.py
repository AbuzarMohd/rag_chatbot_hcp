from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq


def create_rag_chain(retriever, api_key):
    """
    Create the RAG QA chain.
    """

    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
