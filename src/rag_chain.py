from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq


def create_rag_chain(retriever, api_key):

    llm = ChatGroq(
        model="llama3-8b-8192",
        api_key=api_key
    )

    prompt_template = """
You are a college professor helping students understand their study material.

Use ONLY the context provided below to answer the question.

If the answer is not present in the context, say:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain
