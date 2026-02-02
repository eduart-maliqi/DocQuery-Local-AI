import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- SETUP (Pfade & Cache) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_cache_path = os.path.join(current_dir, "models")
os.makedirs(model_cache_path, exist_ok=True)


os.environ["HUGGINGFACE_HUB_CACHE"] = model_cache_path
os.environ["SENTENCE_TRANSFORMERS_HOME"] = model_cache_path

st.set_page_config(page_title="DocQuery", layout="wide")
st.title("🤖 DocQuery: Lokale Vektordatenbank")

# --- FUNKTIONEN ---

@st.cache_resource # Sorgt dafür, dass das Modell nur 1x geladen wird
def get_embeddings_model():
    
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=model_cache_path
    )

def process_pdf(uploaded_file):
    temp_path = os.path.join(current_dir, "temp.pdf")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader(temp_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    os.remove(temp_path)
    return chunks

# --- UI & LOGIK ---

with st.sidebar:
    st.header("Konfiguration")
    hf_token = st.text_input("HuggingFace API Token", type="password")
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

uploaded_files = st.file_uploader("PDF hochladen", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    for file in uploaded_files:
        with st.spinner(f"Lese {file.name}..."):
            all_chunks.extend(process_pdf(file))
    
    st.info(f"Insgesamt {len(all_chunks)} Textabschnitte geladen.")

    # Vektordatenbank erstellen
    if st.button("Wissen in Datenbank speichern"):
        with st.spinner("KI generiert Vektoren (beim ersten Mal Download-Dauer beachten)..."):
            embeddings = get_embeddings_model()
            # Hier entsteht die Datenbank aus den Text-Stücken
            vector_db = FAISS.from_documents(all_chunks, embeddings)
            
            # Wir speichern die DB lokal, damit wir sie nicht jedes Mal neu bauen müssen
            vector_db.save_local(os.path.join(current_dir, "faiss_index"))
            st.success("Datenbank ist bereit! Du kannst jetzt Fragen stellen.")