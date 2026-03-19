import streamlit as st
import validators
import re

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- UI ----------------
st.set_page_config(page_title="AI Summarizer", layout="centered")
st.title("🚀 AI Summarizer (Fast + Production Ready)")
st.write("Summarize YouTube videos or Web pages instantly")

with st.sidebar:
    groq_api_key = st.text_input("🔑 Groq API Key", type="password")

url = st.text_input("Enter URL (YouTube or Website)")

# ---------------- Prompt ----------------
prompt = PromptTemplate.from_template(
    "Summarize the following content in 200–300 words:\n\n{text}"
)

# ---------------- Utils ----------------
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
    return match.group(1) if match else None


@st.cache_data(show_spinner=False)
def load_youtube_data(url):
    video_id = extract_video_id(url)
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in transcript])
    return [Document(page_content=text)]


@st.cache_data(show_spinner=False)
def load_web_data(url):
    loader = WebBaseLoader(url)
    return loader.load()


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


def summarize(llm, docs):
    summaries = []

    # Map step (summarize chunks)
    for doc in docs:
        chain = prompt | llm
        result = chain.invoke({"text": doc.page_content})
        summaries.append(result.content)

    # Reduce step (final summary)
    combined = " ".join(summaries)
    final = (prompt | llm).invoke({"text": combined})

    return final.content


# ---------------- Main Logic ----------------
if st.button("✨ Summarize"):

    if not groq_api_key or not url:
        st.error("Please provide API key and URL")
        st.stop()

    if not validators.url(url):
        st.error("Invalid URL")
        st.stop()

    try:
        with st.spinner("Processing..."):

            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model="llama-3.1-8b-instant"
            )

            # Load data
            if "youtu" in url:
                docs = load_youtube_data(url)
            else:
                docs = load_web_data(url)

            # Split
            docs = split_docs(docs)

            # Summarize
            result = summarize(llm, docs)

            st.success(result)

    except Exception as e:
        st.error("Something went wrong")
        st.exception(e)
