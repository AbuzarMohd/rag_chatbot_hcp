import asyncio
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI


def create_rag_chain(retriever, api_key):

    # ✅ FIX: Ensure event loop exists (important for Streamlit)
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        google_api_key=api_key,
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    return qa_chain
