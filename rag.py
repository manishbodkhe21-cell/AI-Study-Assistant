from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import ollama

# Path to vector database
CHROMA_PATH = "data/chroma"

# Load embedding model
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to vector database
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)

# Create retriever
retriever = vectordb.as_retriever(search_kwargs={"k": 3})


def ask_question(question):

    docs = retriever.invoke(question)

    # If nothing retrieved
    if not docs:
        return "I don't have enough information in the notes to answer that.", []

    context = ""
    sources = []

    for doc in docs:

        context += doc.page_content + "\n\n"

        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        sources.append(f"{src} (Page {page})")

    # If context too small
    if len(context.strip()) < 50:
        return "I don't have enough information in the notes to answer that.", []

    prompt = f"""
You are an AI study assistant.

Only answer using the information provided in the context.

If the context does NOT contain enough information to answer the question,
respond exactly with:

"I don't have enough information in the notes to answer that."

Context:
{context}

Question:
{question}
"""

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    return answer, sources