from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


def create_rag_chain(retriever, api_key):

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
