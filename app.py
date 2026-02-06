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

st.set_page_config(page_title="DocQuery Pro", layout="wide")
st.title("🤖 DocQuery: PDF, Word & Video Chat")

# Session State initialisieren
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- 2. FUNKTIONEN ---

@st.cache_resource
def load_embedder():
    """Lädt das Modell zur Text-Vektorisierung (Bedeutungsverständnis)."""
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)

def extract_text_from_file(uploaded_file, hf_token):
    """Extrahiert Text aus verschiedenen Formaten oder transkribiert MP4."""
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
        with st.spinner(f"🎥 Video wird transkribiert: {filename}..."):
            # Temporäre Dateien speichern
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Audio aus Video extrahieren mit VideoFileClip
            video = VideoFileClip("temp_video.mp4")
            video.audio.write_audiofile("temp_audio.mp3", logger=None)
            
            # KI-Transkription (Whisper via HuggingFace)
            client = InferenceClient(api_key=hf_token)
            audio_result = client.automatic_speech_recognition("temp_audio.mp3")
            text = audio_result["text"]
            
            # Aufräumen
            video.close()
            if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")
            if os.path.exists("temp_audio.mp3"): os.remove("temp_audio.mp3")

    return text

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Konfiguration")
    if "hf_token" not in st.session_state:
        st.session_state.hf_token = ""
    
    token_input = st.text_input("HuggingFace Token", value=st.session_state.hf_token, type="password")
    if token_input:
        st.session_state.hf_token = token_input

    st.divider()
    if st.button("Chat-Verlauf löschen"):
        st.session_state.messages = []
        st.rerun()

# --- 4. DATEI UPLOAD & VERARBEITUNG ---
uploaded_files = st.file_uploader(
    "Dateien hochladen (PDF, DOCX, MP4)", 
    type=["pdf", "docx", "mp4"], 
    accept_multiple_files=True
)

if uploaded_files and st.session_state.hf_token:
    if st.button("Dokumente analysieren"):
        with st.spinner("Inhalte werden verarbeitet..."):
            all_chunks = []
            for file in uploaded_files:
                raw_text = extract_text_from_file(file, st.session_state.hf_token)
                if raw_text:
                    for i in range(0, len(raw_text), 1000):
                        chunk = f"QUELLE: {file.name}\nINHALT: {raw_text[i:i+1000]}"
                        all_chunks.append(chunk)
            
            st.session_state.chunks = all_chunks
            
            # Embeddings erstellen
            embedder = load_embedder()
            embeddings = embedder.encode(all_chunks)
            
            # FAISS Index (Gedächtnis) bauen
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.vector_index = index
            
            st.success(f"Erfolgreich! {len(all_chunks)} Textstellen gelernt.")

# --- 5. CHAT INTERFACE ---
st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Frag mich etwas..."):
    if not st.session_state.hf_token or "vector_index" not in st.session_state:
        st.error("Bitte Token eingeben und Dateien verarbeiten!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Suche läuft..."):
                embedder = load_embedder()
                query_vec = embedder.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_vec).astype('float32'), k=3)
                
                context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
                
                # RAG Antwort generieren
                client = InferenceClient(api_key=st.session_state.hf_token)
                system_instr = f"Antworte nur basierend auf diesem Kontext: {context}"
                
                try:
                    full_response = ""
                    for message in client.chat_completion(
                        model="mistralai/Mistral-7B-Instruct-v0.2",
                        messages=[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}],
                        max_tokens=800,
                        stream=True
                    ):
                        full_response += message.choices[0].delta.content or ""
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Fehler: {e}")