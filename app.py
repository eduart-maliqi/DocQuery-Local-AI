import streamlit as st

# Titel der App
st.set_page_config(page_title="DocQuery - Lokale RAG App", layout="wide")
st.title("🤖 DocQuery: Chatte mit deinen Dokumenten")

# Seitenleiste für Einstellungen
with st.sidebar:
    st.header("Einstellungen")
    hf_token = st.text_input("HuggingFace API Token", type="password")
    st.info("Dein Token wird nur lokal für diese Session verwendet.")

# Datei-Uploader
uploaded_files = st.file_uploader(
    "Lade PDFs, Word-Dateien oder MP4-Videos hoch", 
    type=["pdf", "docx", "mp4"], 
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} Datei(en) erfolgreich hochgeladen!")