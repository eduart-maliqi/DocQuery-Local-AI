import os
import streamlit as st
from pypdf import PdfReader
import docx
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai
from google.genai import types

# --- 1. SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models")
os.makedirs(model_path, exist_ok=True)

# I am setting up the English UI for the application
st.set_page_config(page_title="DocQuery Gemini Pro", layout="wide")
st.title("♊ DocQuery: Hybrid Search (Files + Web)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- 2. FUNCTIONS ---

@st.cache_resource
def load_embedder():
    # I am loading the local embedding model for the RAG process
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)

def extract_text_from_file(uploaded_file, gemini_key):
    # I am handling different file types to extract the raw text
    filename = uploaded_file.name
    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    elif filename.endswith(".mp4"):
        # I am processing video files by extracting and transcribing audio
        with st.spinner(f"🎥 Processing video: {filename}..."):
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            video = VideoFileClip("temp_video.mp4")
            video.audio.write_audiofile("temp_audio.mp3", logger=None)
            
            # I am leaving a placeholder for transcription logic
            text = "Video transcription placeholder..." 
            video.close()
            os.remove("temp_video.mp4")
            os.remove("temp_audio.mp3")
    return text

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    # I am storing the Gemini API Key in the session state
    gemini_key = st.text_input("Gemini API Key", type="password")
    if gemini_key:
        st.session_state.gemini_key = gemini_key
    
    # I am adding a model selector to switch if one model hits a limit
    model_choice = st.selectbox("Select Model", 
                                ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash"],
                                help="Use 1.5-flash if 2.0 shows 'Limit 0' errors")

    st.divider()
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. DATA PROCESSING ---
uploaded_files = st.file_uploader("Upload Files (PDF, DOCX, MP4)", type=["pdf", "docx", "mp4"], accept_multiple_files=True)

if uploaded_files and "gemini_key" in st.session_state:
    if st.button("Index Documents"):
        with st.spinner("I am analyzing the documents..."):
            all_chunks = []
            for file in uploaded_files:
                raw_text = extract_text_from_file(file, st.session_state.gemini_key)
                if raw_text:
                    for i in range(0, len(raw_text), 1500):
                        # I am including the filename as metadata for the AI
                        chunk = f"SOURCE: {file.name}\nCONTENT: {raw_text[i:i+1500]}"
                        all_chunks.append(chunk)
            
            st.session_state.chunks = all_chunks
            embedder = load_embedder()
            embeddings = embedder.encode(all_chunks)
            
            # I am using FAISS for high-speed local vector search
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.vector_index = index
            st.success("I have indexed all documents successfully!")

# --- 5. INTERACTIVE CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me anything..."):
    if not st.session_state.get("gemini_key") or "vector_index" not in st.session_state:
        st.error("I need an API key and indexed documents to help you!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("I am checking my records and searching the web..."):
                # I am retrieving the most relevant chunks from your files
                embedder = load_embedder()
                query_vec = embedder.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_vec).astype('float32'), k=5)
                context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
                
                # I am initializing the Gemini client with the Search tool
                client = genai.Client(api_key=st.session_state.gemini_key)
                
                # I am using your custom system prompt instructions
                full_prompt = f"""
                You are an expert assistant. Use the following context to answer the question.
                If the answer is not in the context, search for what the answer would be but inform the user 
                that it is not in the files but that you got it from the internet and state exactly which 
                source you used for the answer. If you cannot answer the question, say so.
                
                CONTEXT:
                {context}
                
                QUESTION:
                {prompt}
                """
                
                try:
                    # I am performing the hybrid search using Gemini and Google Search
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=full_prompt,
                        config=types.GenerateContentConfig(
                            tools=[types.Tool(google_search=types.GoogleSearch())]
                        )
                    )
                    
                    # I am extracting and displaying the generated response
                    answer = response.text
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    # I am displaying an error if the API limit or quota is hit
                    st.error(f"Gemini API Error: {e}")