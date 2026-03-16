def get_retriever(vectorstore):
    """
    Convert vector store into a retriever.
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    return retriever
