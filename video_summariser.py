import validators
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.schema import Document
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
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])


def extract_video_id(url):
    """Extract YouTube video ID from various URL formats."""
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
    """Fetch transcript using YouTubeTranscriptApi directly."""
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Could not extract video ID from URL.")

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)

    full_text = " ".join([entry.text for entry in transcript_list])
    return [Document(page_content=full_text)]


if st.button("Summarise the content from YT or Website"):
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL")

    else:
        try:
            with st.spinner("Waiting..."):
                llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model="llama-3.1-8b-instant"
                )

                if "youtube.com" in generic_url or "youtu.be" in generic_url:
                    try:
                        docs = get_youtube_transcript(generic_url)

                        if not docs:
                            st.error("No transcript could be loaded for this YouTube video.")
                            st.stop()

                    except Exception as yt_error:
                        st.error(f"Could not fetch YouTube transcript: {yt_error}")
                        st.stop()

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0"}
                    )
                    docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)

        except Exception as e:
            st.exception(e)
