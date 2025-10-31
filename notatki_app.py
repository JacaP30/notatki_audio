from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from streamlit_option_menu import option_menu
from streamlit.runtime.secrets import StreamlitSecretNotFoundError
from pathlib import Path
from qdrant_client.http.exceptions import UnexpectedResponse


# Configuration & Env
env = dotenv_values(".env")
#=======================================================
# Ta część jest zakomentowana, bo teraz lokalnie używamy st.secrets
# Do wdrażania trzeba odkomentować
# Load Qdrant credentials from Streamlit Secrets if present
try:
    if 'QDRANT_URL' in st.secrets:
        env['QDRANT_URL'] = st.secrets['QDRANT_URL']
    if 'QDRANT_API_KEY' in st.secrets:
        env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
    if 'OPENAI_API_KEY' in st.secrets:
        env['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
except StreamlitSecretNotFoundError:
    # Brak pliku secrets.toml – pracujemy na wartościach z .env / lokalnych
    pass
#=======================================================





# Wybór embedowania zgodny z istniejącą kolekcją (3072)
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIM = 3072

AUDIO_TRANSCRIBE_MODEL = "whisper-1"
QDRANT_COLLECTION_NAME = "notes"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def get_qdrant_client():
    url = env.get("QDRANT_URL")
    api_key = env.get("QDRANT_API_KEY")
    if not url or not api_key:
        raise RuntimeError("Missing QDRANT_URL and QDRANT_API_KEY in environment or secrets.")
    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_qdrant_client_cached():
    # Prefer cached wrapper if you want to reuse
    return get_qdrant_client()

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client_cached()
    try:
        if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
            print("Tworzę kolekcję")
            qdrant_client.create_collection(
                collection_name=QDRANT_COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
        else:
            print("Kolekcja już istnieje")
    except UnexpectedResponse as e:
        st.error(
            "Nie mogę połączyć się z Qdrant (404 Not Found). Sprawdź poprawność QDRANT_URL (np. 'http://localhost:6333' lub pełny REST URL z Qdrant Cloud) oraz klucza API.\n\n"
            f"Aktualny QDRANT_URL: {env.get('QDRANT_URL', '(brak)')}\n\nSzczegóły: {e}")
        st.stop()

def init_openai_key_if_needed():
    if not st.session_state.get("openai_api_key"):
        if "OPENAI_API_KEY" in env:
            st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
        else:
            st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
            st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
            if st.session_state["openai_api_key"]:
                st.rerun()
    if not st.session_state.get("openai_api_key"):
        st.stop()

def get_embeddings(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(

        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    try:
        return result.data[0].embedding
    except Exception:
        if isinstance(result, dict) and "data" in result and isinstance(result["data"], list) and len(result["data"]) > 0:
            return result["data"][0].get("embedding")
        raise

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    audio_file.seek(0)
    try:
        transcript = openai_client.audio.transcriptions.create(
            file=audio_file,
            model=AUDIO_TRANSCRIBE_MODEL,
            response_format="verbose_json",
        )
    except Exception as e:
        st.error(f"Błąd transkrypcji audio: {e}")
        return ""
    text = None
    if isinstance(transcript, dict):
        text = transcript.get("text")
    else:
        text = getattr(transcript, "text", None)
        if text is None and hasattr(transcript, "data"):
            data = getattr(transcript, "data")
            if isinstance(data, dict):
                text = data.get("text")
            elif isinstance(data, list) and len(data) > 0:
                t0 = data[0]
                text = getattr(t0, "text", None) or (t0.get("text") if isinstance(t0, dict) else None)
    return text if isinstance(text, str) else ""

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client_cached()
    import time

    # Używamy timestamp jako ID aby uniknąć konfliktów
    note_id = int(time.time() * 1000)  # milliseconds timestamp
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=note_id,
                vector=get_embeddings(text=note_text),
                payload={
                    "text": note_text,
                    "created_at": note_id,  # zapisujemy timestamp do sortowania
                },
            )
        ]
    )

def delete_note_from_db(note_id):
    qdrant_client = get_qdrant_client_cached()
    qdrant_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=[note_id],
    )

def list_notes_from_db(query=None):
    """Pobiera notatki: wszystkie (bez query) lub semantycznie (z query)"""
    qdrant_client = get_qdrant_client_cached()

    if not query:
        notes = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )[0]

        result = []
        for note in notes:
            payload = note.payload or {}
            result.append({
                "id": note.id,
                "text": payload.get("text", ""),
                "created_at": payload.get("created_at", 0),
                "score": None,
            })
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result
    else:
        # Wyszukiwanie semantyczne
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embeddings(text=query),

            limit=100,
            with_payload=True,
        )

        result = []
        for note in notes:
            payload = note.payload or {}
            result.append({
                "id": note.id,
                "text": payload.get("text", ""),
                "created_at": payload.get("created_at", 0),
                "score": note.score,
            })
        return result
    
#=======================================================
# MAIN
st.set_page_config(
    page_title="Audio Notatki",
    page_icon="🎤",
    layout="centered",
    menu_items={"Get help": None, "Report a bug": None, "About": None}
)

HIDE_STREAMLIT_STYLE = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}
    .stDeployButton, [data-testid="stBaseButton-header"], [data-testid="stDecoration"] {display: none !important;}

    /* PRAWY DOLNY RÓG — Manage app (Cloud) */
    button[aria-label="Manage app"] {display: none !important;}
    a[aria-label="Manage app"] {display: none !important;}
    [data-testid="manageAppButton"] {display: none !important;}
    [data-testid="stCloudManageApp"] {display: none !important;}
    div[role="complementary"] [title="Manage app"] {display: none !important;}
    </style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# OpenAI API key protection
init_openai_key_if_needed()
if not st.session_state.get("openai_api_key"):
    st.stop()

# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

# Główna część aplikacji
column1, column2 = st.columns([2,6])
with column1:
    logo_path = Path("logo.png")
    if logo_path.exists():
        st.image(str(logo_path), width=120)
    else:
        st.markdown("<div style='font-size: 56px; line-height: 1;'>🎤</div>", unsafe_allow_html=True)
with column2:
    st.markdown("<h1 style='background: linear-gradient(130deg, #eb2a91ff 25%, #1567eaff 60%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎤 Audio Notatki 📝</h1>", unsafe_allow_html=True)
st.markdown("""
Aplikacja do tworzenia szybkich notatek z transkrypcji audio, z zapisem w bazie danych Qdrant.<br>
Wyszukiwanie w zapisanych działa semantycznie z wykorzystaniem OpenAI.
""", unsafe_allow_html=True)

db_configured = bool(env.get("QDRANT_URL") and env.get("QDRANT_API_KEY"))

if db_configured:
    assure_db_collection_exists()
else:
    st.warning("Konfiguracja Qdrant nie ustawiona. Aby zapisywać notatki, ustaw QDRANT_URL i QDRANT_API_KEY w .env lub Secrets.")

selected = option_menu(None, ["Dodaj notatkę", "Wyszukaj notatkę"],
    icons=['record', 'search'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected == "Dodaj notatkę":
    note_audio = audiorecorder(
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        if st.button("🖋️Transkrybuj audio"):
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])

        if st.session_state["note_text"] and st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            add_note_to_db(note_text=st.session_state["note_text"])
            st.toast("Notatka zapisana", icon="💾")
            st.session_state["note_text"] = ""
            st.session_state["note_audio_text"] = ""
            st.session_state["note_audio_bytes"] = None
            st.session_state["note_audio_bytes_md5"] = None
            st.rerun()
elif selected == "Wyszukaj notatkę":
    query = st.text_input("Wyszukaj notatkę")

    # Przycisk wyszukiwania
    search_clicked = st.button("Szukaj")

    if not db_configured:
        st.warning("Konfiguracja Qdrant nie ustawiona. Aby przeszukiwać notatki, ustaw QDRANT_URL i QDRANT_API_KEY w .env lub Secrets.")
    else:
        if search_clicked or not query:
            notes = list_notes_from_db(query)
            if not notes:
                st.info("Nie znaleziono żadnych notatek")
            else:
                for note in notes:
                    with st.container():
                        col1, col2 = st.columns([5,1])
                        with col1:
                            st.markdown(note["text"])
                            if note["score"]:
                                st.markdown(f':violet[{note["score"]}]')
                        with col2:
                            if st.button("🗑️", key=f"delete_{note['id']}", help="Usuń notatkę"):
                                delete_note_from_db(note["id"])
                                st.toast("Notatka usunięta", icon="🗑️")
                                st.rerun()
                                