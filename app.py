import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from huggingface_hub import InferenceClient

# --- SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models")
os.makedirs(model_path, exist_ok=True)

st.set_page_config(page_title="DocQuery Light", layout="wide")
st.title("🚀 DocQuery: Stelle Fragen zu deinem PDF")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- FUNKTIONEN ---

@st.cache_resource
def load_embedder():
    # Wir laden das Modell direkt über SentenceTransformers
    return SentenceTransformer("all-MiniLM-L6-v2", cache_folder=model_path)

def get_pdf_text(pdf_files):
    text_chunks = []
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        # Wir merken uns hier den Namen der Datei
        filename = pdf.name 
        
        for page_num, page in enumerate(reader.pages):
            content = page.extract_text()
            if content:
                # Wir schneiden den Text in Stücke
                for i in range(0, len(content), 1000):
                    # WICHTIG: Wir schreiben den Dateinamen direkt ÜBER den Textabschnitt
                    chunk_with_metadata = f"DATEI: {filename} | SEITE: {page_num+1}\n\n{content[i:i+1000]}"
                    text_chunks.append(chunk_with_metadata)
    return text_chunks

# --- UI & LOGIK ---

with st.sidebar:
    hf_token = st.text_input("HuggingFace Token", type="password")
    if st.button("Chat löschen"):
        st.session_state.messages = []
        st.rerun()

uploaded_files = st.file_uploader("PDFs hochladen", type=["pdf"], accept_multiple_files=True)

if uploaded_files and hf_token:
    if st.button("Datenbank aufbauen"):
        with st.spinner("Verarbeite Dokumente..."):
            # 1. Text extrahieren
            chunks = get_pdf_text(uploaded_files)
            st.session_state.chunks = chunks
            
            # 2. Embeddings erstellen
            model = load_embedder()
            embeddings = model.encode(chunks)
            
            # 3. FAISS Index erstellen
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings).astype('float32'))
            
            st.session_state.vector_index = index
            st.success(f"{len(chunks)} Textstücke gespeichert!")

# --- CHAT ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Frag mich was..."):
    if not hf_token or "vector_index" not in st.session_state:
        st.error("Bitte Token eingeben und PDFs verarbeiten!")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # A. Suche (Retrieval)
            model = load_embedder()
            question_embedding = model.encode([prompt])
            D, I = st.session_state.vector_index.search(np.array(question_embedding).astype('float32'), k=3)
            
            # Die gefundenen Texte zusammenfügen
            context = "\n".join([st.session_state.chunks[i] for i in I[0]])
            
            # B. KI fragen (Generation)
            client = InferenceClient(api_key=hf_token)
            system_prompt = f"Nutze diesen Text um die Frage zu beantworten: {context}"
            
            response = ""
            for message in client.chat_completion(
                model="mistralai/Mistral-7B-Instruct-v0.2",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                max_tokens=500,
                stream=True,
            ):
                response += message.choices[0].delta.content or ""
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})