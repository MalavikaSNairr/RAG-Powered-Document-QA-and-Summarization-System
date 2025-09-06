import os
import streamlit as st
import traceback
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
import torch

# Detect device GPU if available else CPU
DEVICE = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_embeddings():
    # Embedding model with moderate size and fast inference
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorstore(file_path, _embeddings):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        loader = PyPDFLoader(file_path)
    documents = loader.load()
    # Small chunk size and overlap to fit model max token limits 
    splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, _embeddings)

@st.cache_resource
def load_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=DEVICE,
        truncation=True,
        max_length=512,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)

def ask_question(vectorstore, query, k=5):
    try:
        llm = load_llm()
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": k}),
            chain_type="map_reduce"
        )
        result = qa.run(query)
        if not result or not result.strip():
            return " No answer found."
        return result
    except Exception as e:
        tb = traceback.format_exc()
        print("Error ask_question:\n", tb)
        return f" {str(e)}"

def generate_faqs(vectorstore, max_chunks=7):
    llm = load_llm()
    retriever = vectorstore.as_retriever()
    faqs = []
    docs = retriever.get_relevant_documents("")[:max_chunks]
    for doc in docs:
        prompt = (
            "Generate a detailed FAQ question and answer based on the text below.\n"
            "Format as:\nQ: ...\nA: ...\n\n"
            f"Text:\n{doc.page_content}"
        )
        try:
            faq = llm(prompt)
            faqs.append(faq)
        except Exception as e:
            faqs.append(f"Error generating FAQ: {e}")
    return faqs

def summarize_document(vectorstore, max_chunks=5):
    llm = load_llm()
    docs = vectorstore.as_retriever().get_relevant_documents("")[:max_chunks]
    combined_text = "\n\n".join([doc.page_content for doc in docs])
    prompt = (
        "Read the following text and write a concise, logically structured summary. "
        "Combine information and avoid repetition. Highlight key points succinctly:\n\n"
        f"Text:\n{combined_text}"
    )
    try:
        summary = llm(prompt)
        return summary
    except Exception as e:
        return f"Error during summarization: {e}"

# Streamlit app UI
st.set_page_config(page_title="📚 RAG Q&A System", layout="wide")
st.title("📚RAG Document Q&A with Robust Summarization")

uploaded_file = st.file_uploader("Upload TXT or PDF files", type=["txt", "pdf"])

if uploaded_file:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Document processed and indexed!")

    embeddings = load_embeddings()
    vectorstore = load_vectorstore(file_path, embeddings)

    st.subheader("Frequently Asked Questions (FAQs)")
    if st.button("Generate FAQs"):
        with st.spinner("Generating FAQs..."):
            faqs = generate_faqs(vectorstore)
            for faq in faqs:
                st.markdown(f"**{faq}**")

    if st.button("Summarize this document"):
        with st.spinner("Summarizing..."):
            summary = summarize_document(vectorstore)
            st.info(summary)

    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner("Thinking..."):
            answer = ask_question(vectorstore, question)
            if answer.startswith("❌") or answer.startswith("⚠️"):
                st.error(answer)
            else:
                st.write("**Answer:**", answer)
