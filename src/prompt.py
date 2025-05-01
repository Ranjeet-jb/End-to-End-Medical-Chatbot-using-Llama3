from store_index import docsearch

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":5})

retrieved_docs = retriever.invoke("HIV")

# Join the retrieved documents into a single string
retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

# Prepare the system prompt for the model
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "Use three sentences maximum and keep the answer concise."
    "\n\n"
    f"{retrieved_text}"
)
