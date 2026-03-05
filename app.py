import streamlit as st
import os
from ocr import extract_text_from_pdf
from ingest import process_ocr_to_vectordb
from rag import ask_question

st.title("AI Study Assistant")

UPLOAD_FOLDER = "data/pdfs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------- PDF Upload --------
uploaded_file = st.file_uploader("Upload your PDF notes", type=["pdf"])

if uploaded_file:

    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF uploaded successfully")

    st.write("Running OCR...")

    extract_text_from_pdf(file_path)

    st.write("Creating knowledge base...")

    process_ocr_to_vectordb()

    st.success("PDF processed successfully")


# -------- Ask Question --------
question = st.text_input("Ask a question from your notes")

if question:

    answer, sources = ask_question(question)

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Sources")

    for s in sources:
        st.write(s)