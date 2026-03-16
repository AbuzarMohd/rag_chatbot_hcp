import streamlit as st

from src.pdf_loader import load_pdf
from src.text_splitter import split_documents
from src.vector_store import create_vector_store
from src.retriever import get_retriever
from src.rag_chain import create_rag_chain


st.title("PDF RAG Chatbot")

api_key = st.secrets["GROQ_API_KEY"]

documents = load_pdf("data/documents.pdf")

chunks = split_documents(documents)

vectorstore = create_vector_store(chunks, api_key)

retriever = get_retriever(vectorstore)

qa_chain = create_rag_chain(retriever, api_key)

query = st.text_input("Ask a question from the PDF")

if query:
    response = qa_chain.run(query)
    st.write(response)
