import asyncio
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

def create_rag_chain(retriever, api_key):
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=api_key,
    temperature=0.3,
    max_retries=1  # ✅ prevents burning quota on retries
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
