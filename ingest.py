import os
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

OCR_FOLDER = "data/ocr_text"
CHROMA_PATH = "data/chroma"

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def process_ocr_to_vectordb():

    documents = []
    metadatas = []

    for file in os.listdir(OCR_FOLDER):

        if file.endswith(".json"):

            with open(os.path.join(OCR_FOLDER, file), "r") as f:
                data = json.load(f)

            for page in data:

                documents.append(page["text"])

                metadatas.append({
                    "source": page["file"],
                    "page": page["page"]
                })

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = []
    chunk_meta = []

    for doc, meta in zip(documents, metadatas):

        split_chunks = splitter.split_text(doc)

        for chunk in split_chunks:
            chunks.append(chunk)
            chunk_meta.append(meta)

    vectordb = Chroma.from_texts(
        texts=chunks,
        metadatas=chunk_meta,
        embedding=embedding,
        persist_directory=CHROMA_PATH
    )

    vectordb.persist()