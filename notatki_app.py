from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from streamlit_option_menu import option_menu 

 
env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 3072

AUDIO_TRANSCRIBE_MODEL = "whisper-1"
  
QDRANT_COLLECTION_NAME = "notes"
 
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",
    )

    return transcript.text

#
# DB
#
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)

def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
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

def get_embeddings(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
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
# ----------
def delete_note_from_db(note_id):
    qdrant_client = get_qdrant_client()
    qdrant_client.delete(
        collection_name=QDRANT_COLLECTION_NAME,
        points_selector=[note_id],
    )

# ----------
def list_notes_from_db(query=None):
    """Funkcja pobierająca notatki z bazy bez cachowania"""
    qdrant_client = get_qdrant_client()
    
    if not query:
        # Pobierz wszystkie notatki i posortuj je po czasie utworzenia (od najnowszych)
        notes = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=100,  # Zwiększamy limit
            with_payload=True,
            with_vectors=False,
        )[0]
        
        # Sortujemy notatki po czasie utworzenia (od najnowszych)
        result = []
        for note in notes:
            result.append({
                "id": note.id,
                "text": note.payload["text"] if note.payload and "text" in note.payload else "",
                "created_at": note.payload.get("created_at", 0),
                "score": None,
            })
        
        # Sortuj po czasie utworzenia (od najnowszych)
        result.sort(key=lambda x: x["created_at"], reverse=True)
        return result

    else:
        # Wyszukiwanie semantyczne
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embeddings(text=query),
            limit=100,  # Zwiększamy limit
            with_payload=True,
        )
        
        result = []
        for note in notes:
            result.append({
                "id": note.id,
                "text": note.payload["text"] if note.payload and "text" in note.payload else "",
                "created_at": note.payload.get("created_at", 0),
                "score": note.score,
            })
            
        return result


#
# MAIN
#
st.set_page_config(
    page_title="Audio Notatki", 
    page_icon="🎤", 
    layout="centered"
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None,}
    )
HIDE_STREAMLIT_STYLE = """
    <style>
    /* Stare selektory (działają w wielu wersjach) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Nowszy toolbar (prawy górny róg) */
    [data-testid="stToolbar"] {visibility: hidden; height: 0; position: fixed;}

    /* Przyciski deploy/share — różne wersje używają różnych selektorów */
    .stDeployButton {display: none !important;}
    [data-testid="stBaseButton-header"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    </style>
"""
st.markdown(HIDE_STREAMLIT_STYLE, unsafe_allow_html=True)

# OpenAI API key protection
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

# Session state initialization
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""


# Główna część aplikacji
column1, column2, = st.columns([2,6])
with column1:
    # Wyświetl logo na górze aplikacji
    st.image("logo.png", width=120)
with column2:
    st.markdown("<h1 style='background: linear-gradient(130deg, #eb2a91ff 25%, #1567eaff 60%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>🎤 Audio Notatki 📝</h1>", unsafe_allow_html=True)
st.markdown("""
Aplikacja do tworzenia szybkich notatek z transkrypcji audio, z zapisem w bazie danych Qdrant.<br>
Wyszukiwanie w zapisanych działa semantycznie z wykorzystaniem OpenAI.
""", unsafe_allow_html=True)

assure_db_collection_exists()

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
            # Wyczyść pola po zapisaniu
            st.session_state["note_text"] = ""
            st.session_state["note_audio_text"] = ""
            st.session_state["note_audio_bytes"] = None
            st.session_state["note_audio_bytes_md5"] = None
            st.rerun()
elif selected == "Wyszukaj notatkę":
    query = st.text_input("Wyszukaj notatkę")
    
    # Przycisk wyszukiwania
    search_clicked = st.button("Szukaj")
    
    # Pobierz notatki jeśli kliknięto szukaj lub pole wyszukiwania jest puste
    if search_clicked or not query:
        notes = list_notes_from_db(query)
        if not notes:
            st.info("Nie znaleziono żadnych notatek")
        else:
            for note in notes:
                with st.container(border=True):
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