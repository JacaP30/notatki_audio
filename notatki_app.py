from io import BytesIO
import streamlit as st
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5 
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

#--------------------------------------------------------------
# Konfiguracja aplikacji
#====================================================================
# Te ustawienia konfigurują aplikację, w tym klucz API OpenAI, model osadzenia, 
# model transkrypcji audio i nazwę kolekcji w bazie danych Qdrant.
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

QDRANT_COLLECTION_NAME = "notes"

AUDIO_TRANSCRIBE_MODEL = "whisper-1"

def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json"
    )

    return transcript.text
#
# Funkcje do obsługi bazy danych Qdrant
#====================================================================
# Te funkcje pozwalają na tworzenie kolekcji w bazie danych Qdrant, 
# dodawanie notatek do bazy danych oraz wyszukiwanie notatek.

# Funkcja do uzyskiwania klienta Qdrant
@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
    ) # path=":memory:" lub QdrantClient(url=env["QDRANT_URL"], api_key=env["QDRANT_API_KEY"])

# Funkcja sprawdzająca istnienie kolekcji i tworząca ją, jeśli nie istnieje
#====================================================================
# Ta funkcja sprawdza, czy kolekcja o nazwie QDRANT_COLLECTION_NAME istnieje w bazie danych Qdrant.
# Jeśli nie istnieje, tworzy ją z odpowiednimi parametrami wektorów.
# Funkcja ta jest wywoływana przy starcie aplikacji, aby zapewnić, że kolekcja jest gotowa do użycia.

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
# Funkcje do obsługi notatek
#====================================================================
# Te funkcje pozwalają na uzyskiwanie wektora osadzenia dla tekstu, 
# dodawanie notatek do bazy danych oraz wyszukiwanie notatek w bazie danych.
def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )

    return result.data[0].embedding

# Funkcja do dodawania notatki do bazy danych
#====================================================================
# Ta funkcja dodaje notatkę do bazy danych Qdrant. 
# Przyjmuje tekst notatki jako argument, oblicza wektor osadzenia dla tego tekstu 
# i zapisuje go w kolekcji QDRANT_COLLECTION_NAME wraz z tekstem notatki.
def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=points_count.count + 1,
                vector=get_embedding(text=note_text),
                payload={
                    "text": note_text,
                },
            )
        ]
    )

# Funkcja do listowania notatek z bazy danych
#====================================================================
# Ta funkcja zwraca listę notatek z bazy danych Qdrant. 
# Jeśli podano zapytanie, wyszukuje notatki na podstawie wektora osadzenia zapytania.
# W przeciwnym razie zwraca ostatnie 10 notatek.
def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        result = []
        for note in notes:
            text = note.payload["text"] if note.payload and "text" in note.payload else ""
            result.append({
                "text": text,
                "score": None,
            })

        return result

    # Jeśli podano zapytanie, wyszukujemy notatki na podstawie wektora osadzenia zapytania
    #====================================================================
    # Ta część kodu wyszukuje notatki w bazie danych Qdrant na podstawie wektora osadzenia zapytania.
    # Zwraca listę notatek z ich tekstem i wynikiem wyszukiwania.   
    else:
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            text = note.payload["text"] if note.payload and "text" in note.payload else ""
            result.append({
                "text": text,
                "score": note.score,
            })

        return result


# GŁÓWNA APLIKACJA
#====================================================================
# Ustawienia strony Streamlit
#====================================================================
# Te ustawienia konfigurują tytuł strony, ikonę i układ aplikacji Streamlit.

st.set_page_config(page_title="Notatki audio",page_icon=":microphone:",layout="centered",)#initial_sidebar_state="expanded")

# Ochrona klucza API OpenAI
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

# Sprawdzenie, czy klucz API OpenAI jest ustawiony
#====================================================================
# Jeśli klucz API OpenAI nie jest ustawiony, aplikacja zostaje zatrzymana.
# Użytkownik musi wprowadzić swój klucz API, aby kontynuować.
if not st.session_state.get("openai_api_key"):
    st.stop()

# Inicjalizacja sesji Streamlit
#====================================================================
# Te zmienne sesji przechowują stan aplikacji, takie jak audio notatki, tekst notatek i ich MD5.
# Są one używane do przechowywania danych między interakcjami użytkownika.  
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

# Sprawdzenie istnienia kolekcji w bazie danych
#====================================================================
# Ta funkcja jest wywoływana przy starcie aplikacji, aby zapewnić, że kolekcja jest gotowa do użycia.
# Jeśli kolekcja nie istnieje, zostanie utworzona z odpowiednimi parametrami wektorów.
st.markdown(
    """
    <h1 style='
        text-align: center; 
        font-size: 2.8rem; 
        font-weight: bold; 
        letter-spacing: 1px;
        background: linear-gradient(90deg, #4F8BF9 0%, #F97C4F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        padding-bottom: 0.5rem;
    '>
        🎙️ Notatki audio z transkrypcją    i wyszukiwaniem semantycznym
    </h1>
    """,
    unsafe_allow_html=True
)
assure_db_collection_exists()


# Główna logika aplikacji
#====================================================================
# Ta część aplikacji obsługuje interakcje użytkownika, takie jak nagrywanie audio, transkrypcja audio i wyszukiwanie notatek.
# Użytkownik może nagrać notatkę audio, transkrybować ją do tekstu i zapisać ją w bazie danych.
# Użytkownik może również wyszukiwać notatki na podstawie tekstu.
add_tab, search_tab = st.tabs(["🎙️ Nagraj notatkę", "🔍 Wyszukaj w notatkach"], width="stretch")
# Streamlit nie pozwala na bezpośrednią konfigurację wyglądu przycisków (np. kolorów, rozmiaru) przez parametry funkcji st.button.
# Możesz jednak użyć komponentu st.markdown z HTML/CSS oraz unsafe_allow_html=True, aby stylizować przyciski, lub użyć bibliotek zewnętrznych (np. streamlit-extras).
# Przykład niestandardowego przycisku:
# st.markdown('<button style="background-color: #4CAF50; color: white; padding: 10px 24px; border: none; border-radius: 4px;">Niestandardowy przycisk</button>', unsafe_allow_html=True)
with add_tab:
    note_audio = audiorecorder(
        start_prompt= "⏺️ Nagraj swoją notatkę",
        stop_prompt="🔴 Zatrzymaj nagrywanie",#icon_size=100,

    )
    # Sprawdzenie, czy nagrano audio
    #====================================================================
    # Jeśli nagrano audio, jest ono konwertowane do formatu MP3 i przechowywane w sesji.
    # Następnie sprawdzany jest MD5 tego audio, aby upewnić się, że nie zostało ono zmienione.
    # Jeśli MD5 się zmieni, tekst notatki jest resetowany.
    # Użytkownik może odsłuchać nagranie, transkrybować je do tekstu i edytować notatkę przed zapisaniem jej w bazie danych.
    if note_audio:
        audio = BytesIO()
        note_audio.export(audio, format="mp3")
        st.session_state["note_audio_bytes"] = audio.getvalue()

        curent_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()

        # Sprawdzenie, czy MD5 audio się zmienił
        #====================================================================
        # Jeśli MD5 audio jest inny niż poprzednio zapisany, 
        # resetujemy tekst notatki i ustawiamy nowy MD5.
        # Dzięki temu użytkownik może nagrać nowe audio i transkrybować je bez problemów.
        if st.session_state["note_audio_bytes_md5"] != curent_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = curent_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

        # Transkrypcja audio
        #====================================================================
        # Jeśli użytkownik kliknie przycisk "Transkrybuj audio", 
        # audio jest transkrybowane do tekstu za pomocą modelu Whisper
        if st.button("🖋️ Transkrybuj audio"):
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

        # Wyświetlenie transkrybowanego tekstu
        #====================================================================
        # Jeśli transkrybowany tekst jest dostępny, 
        # jest on wyświetlany w polu tekstowym, gdzie użytkownik może go edytować.
        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])
        # Umożliwienie użytkownikowi edytowania notatki         
        if st.session_state["note_text"] and st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            add_note_to_db(note_text=st.session_state["note_text"])
            st.toast("Notatka zapisana", icon="💾")

# Wyszukiwanie notatek
#====================================================================
# Ta część aplikacji umożliwia użytkownikowi wyszukiwanie notatek na podstawie tekstu.
# Użytkownik może wpisać zapytanie w polu tekstowym i kliknąć przycisk "Szukaj", 
# aby zobaczyć wyniki wyszukiwania. Wyniki są wyświetlane w kontenerach z pogrubionym tekstem i ewentualnym wynikiem wyszukiwania.  
with search_tab:
    query = st.text_input("", placeholder="Wpisz kontekst wyszukiwania", label_visibility="collapsed")
    if st.button("Szukaj"):
        for note in list_notes_from_db(query):
            with st.container(border=True):
                st.markdown(note["text"])
                if note["score"]:
                    st.markdown(f':violet[{note["score"]}]')

