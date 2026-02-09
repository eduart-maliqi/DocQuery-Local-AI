import os
import streamlit as st
from pypdf import PdfReader
import docx
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient

# --- 1. SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models")
os.makedirs(model_path, exist_ok=True)


st.set_page_config(page_title="DocQuery HF Edition", layout="wide")
st.title("🤖 DocQuery: Powered by Hugging Face")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- 2. FUNCTIONS ---

@st.cache_resource
def load_embedder():
    # loading the local embedding model for the search
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)

def extract_text_from_file(uploaded_file, hf_token):
    # extracting text from PDF, Word, or transcribing MP4 via HF
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
        with st.spinner(f"🎥 Transcribing video: {filename}..."):
            #saving the video temporarily
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # extracting the audio using MoviePy
            video = VideoFileClip("temp_video.mp4")
            video.audio.write_audiofile("temp_audio.mp3", logger=None)
            
            # using HF Whisper for the transcription
            client = InferenceClient(api_key=hf_token)
            audio_result = client.automatic_speech_recognition("temp_audio.mp3")
            text = audio_result["text"]
            
            # cleaning up the temporary files
            video.close()
            os.remove("temp_video.mp4")
            os.remove("temp_audio.mp3")

    return text

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Settings")
    #using the HF Token instead of Gemini
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = ""
    
    token_input = st.text_input("Hugging Face Token", value=st.session_state.hf_token, type="password")
    if token_input:
        st.session_state.hf_token = token_input

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 4. DATA PROCESSING ---
uploaded_files = st.file_uploader("Upload Files (PDF, DOCX, MP4)", type=["pdf", "docx", "mp4"], accept_multiple_files=True)

if uploaded_files and st.session_state.hf_token:
    if st.button("Analyze Documents"):
        with st.spinner("I am building the knowledge base..."):
            all_chunks = []
            for file in uploaded_files:
                raw_text = extract_text_from_file(file, st.session_state.hf_token)
                if raw_text:
                    for i in range(0, len(raw_text), 1000):
                        # tagging the source for the AI
                        chunk = f"SOURCE: {file.name}\nCONTENT: {raw_text[i:i+1000]}"
                        all_chunks.append(chunk)
            
            st.session_state.chunks = all_chunks
            embedder = load_embedder()
            embeddings = embedder.encode(all_chunks)
            
            # creating the FAISS index locally
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.vector_index = index
            st.success(f"Success! I learned from {len(all_chunks)} text segments.")

# --- 5. CHAT INTERFACE ---
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask me about your files..."):
    if not st.session_state.hf_token or "vector_index" not in st.session_state:
        st.error("I need your HF Token and indexed documents!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("I am thinking..."):
                # searching for the 3 most relevant context parts
                embedder = load_embedder()
                query_vec = embedder.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_vec).astype('float32'), k=3)
                
                context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
                
                # calling the HF Mistral model
                client = InferenceClient(api_key=st.session_state.hf_token)

                system_instr = (
                    f"You are an expert assistant. Use the following context to answer the question. "
                    f"If the answer is not in the context, use your general knowledge but clearly state "
                    f"that the information is not in the uploaded files. "
                    f"\n\nCONTEXT:\n{context}"
                )
                
                try:
                    full_response = ""
                    for message in client.chat_completion(
                        model="mistralai/Mistral-7B-Instruct-v0.2",
                        messages=[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}],
                        max_tokens=500,
                        stream=True
                    ):
                        full_response += message.choices[0].delta.content or ""
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error: {e}. (Maybe the model is still loading?)")