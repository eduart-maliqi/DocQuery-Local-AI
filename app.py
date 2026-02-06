import os
import streamlit as st
from pypdf import PdfReader
import docx
from moviepy import VideoFileClip
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from google import genai  # Die neue 2026 Bibliothek

# --- 1. SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models")
os.makedirs(model_path, exist_ok=True)

st.set_page_config(page_title="DocQuery Gemini Pro", layout="wide")
st.title("♊ DocQuery: Powered by Gemini")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- 2. FUNKTIONEN ---

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)

def extract_text_from_file(uploaded_file):
    """Extrahiert Text aus PDF oder DOCX."""
    filename = uploaded_file.name
    text = ""
    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        for page in reader.pages:
            text += page.extract_text() or ""
    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])
    return text

# --- 3. SIDEBAR ---
with st.sidebar:
    st.header("Konfiguration")
    gemini_key = st.text_input("Gemini API Key", type="password")
    if gemini_key:
        st.session_state.gemini_key = gemini_key

    st.divider()
    if st.button("Chat löschen"):
        st.session_state.messages = []
        st.rerun()

# --- 4. VERARBEITUNG ---
uploaded_files = st.file_uploader("Dateien hochladen", type=["pdf", "docx"], accept_multiple_files=True)

if uploaded_files and "gemini_key" in st.session_state:
    if st.button("Wissen generieren"):
        with st.spinner("Analysiere Dokumente..."):
            all_chunks = []
            for file in uploaded_files:
                raw_text = extract_text_from_file(file)
                if raw_text:
                    for i in range(0, len(raw_text), 1500): # Größere Chunks für Gemini!
                        chunk = f"QUELLE: {file.name}\n{raw_text[i:i+1500]}"
                        all_chunks.append(chunk)
            
            st.session_state.chunks = all_chunks
            embedder = load_embedder()
            embeddings = embedder.encode(all_chunks)
            
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            st.session_state.vector_index = index
            st.success("Gemini ist bereit!")

# --- 5. CHAT MIT GEMINI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Frag Gemini etwas über deine Dateien..."):
    if not st.session_state.get("gemini_key") or "vector_index" not in st.session_state:
        st.error("Bitte API Key eingeben und Dokumente laden!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Gemini denkt nach..."):
                # 1. Relevante Stellen suchen (RAG)
                embedder = load_embedder()
                query_vec = embedder.encode([prompt])
                D, I = st.session_state.vector_index.search(np.array(query_vec).astype('float32'), k=5)
                
                context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])
                
                # 2. Gemini Client initialisieren
                client = genai.Client(api_key=st.session_state.gemini_key)
                
                # 3. Antwort generieren
                full_prompt = f"""
                Du bist ein Experten-Assistent. Nutze den folgenden Kontext, um die Frage zu beantworten.
                Falls die Antwort nicht im Kontext steht, suche nach was die antwort wäre aber sage dem user
                das es nicht in den files steht sondern das du das vom internet hast und sage genau von welcher
                quelle du die antwort hast. Wenn du die frage nicht beantworten kannst, sag das auch.
                
                KONTEXT:
                {context}
                
                FRAGE:
                {prompt}
                """
                
                try:
                    # ich nutzen Gemini 2.0 Flash
                    response = client.models.generate_content(
                        model="gemini-2.0-flash", 
                        contents=full_prompt
                    )
                    
                    answer = response.text
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Gemini Fehler: {e}")