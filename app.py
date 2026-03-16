import streamlit as st

from src.pdf_loader import load_pdfs
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain


st.title("PDF RAG Chatbot")

# Get Groq API key
api_key = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def initialize_rag():

    documents = load_pdfs("data")

    chunks = split_documents(documents)

    vectorstore = create_vector_store(chunks)

    retriever = get_retriever(vectorstore)

    qa_chain = create_rag_chain(retriever, api_key)

    return qa_chain


qa_chain = initialize_rag()

query = st.text_input("Ask a question from the PDFs")

if query:
    response = qa_chain.run(query)
    st.write(response)
