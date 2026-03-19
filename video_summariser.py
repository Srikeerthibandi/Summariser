import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
import re

st.set_page_config(page_title="LangChain: Summarize Text from YT or Website")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarise URL")

with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

generic_url = st.text_input("URL", label_visibility="collapsed")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""
prompt = PromptTemplate.from_template(prompt_template)


def extract_video_id(url):
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
        r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def get_youtube_transcript(url):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([entry["text"] for entry in transcript])

    return [Document(page_content=text)]


def summarize_docs(llm, docs):
    full_text = " ".join([doc.page_content for doc in docs])
    chain = prompt | llm
    response = chain.invoke({"text": full_text})
    return response.content


if st.button("Summarise the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Processing..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model="llama-3.1-8b-instant"
                )

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    docs = get_youtube_transcript(generic_url)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                summary = summarize_docs(llm, docs)

                st.success(summary)

        except Exception as e:
            st.exception(e)
