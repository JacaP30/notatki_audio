from io import BytesIO
import requests
from datetime import datetime
from urllib.parse import urlparse
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os
import streamlit as st
from audiorecorder import audiorecorder
from dotenv import dotenv_values
from hashlib import md5
from openai import OpenAI
from pydub import AudioSegment
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from streamlit_option_menu import option_menu
from streamlit.errors import StreamlitSecretNotFoundError
from pathlib import Path
from qdrant_client.http.exceptions import UnexpectedResponse

# Configuration & Env
env = dotenv_values(".env")
#=======================================================
# Secrets
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
    
    # Usuń białe znaki (spacje, nowe linie) z wartości
    if url:
        url = url.strip()
    if api_key:
        api_key = api_key.strip()
    
    if not url or not api_key:
        raise RuntimeError("Missing QDRANT_URL and QDRANT_API_KEY in environment or secrets.")
    
    return QdrantClient(url=url, api_key=api_key)

@st.cache_resource
def get_qdrant_client_cached():
    # Prefer cached wrapper if you want to reuse
    return get_qdrant_client()

def assure_db_collection_exists():
    # Debug: sprawdź czy wartości są poprawnie wczytane (bez pokazywania pełnego klucza)
    url_from_env_raw = env.get("QDRANT_URL")
    url_from_env = url_from_env_raw.strip() if isinstance(url_from_env_raw, str) else None
    key_from_env = env.get("QDRANT_API_KEY", "")
    key_preview = f"{key_from_env[:20]}..." if key_from_env and len(key_from_env) > 20 else "(brak)"
    
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
        error_str = str(e).lower()
        error_status = getattr(e, 'status_code', None) or (403 if 'forbidden' in error_str or '403' in str(e) else None) or (404 if '404' in str(e) or 'not found' in error_str else None)
        # obsługa błędów połączenia z Qdrant
        if error_status == 403 or 'forbidden' in error_str:
            st.error(
                "🔒 **Błąd autoryzacji (403 Forbidden)**\n\n"
                "Klucz API jest nieprawidłowy lub nie ma odpowiednich uprawnień.\n\n"
                "**Możliwe przyczyny lokalnie (działa na Streamlit Cloud):**\n"
                "1. Plik `.env` ma białe znaki w wartościach (spacje, nowe linie)\n"
                "2. Cache Streamlit trzyma stare wartości - wyczyść cache i zrestartuj\n"
                "3. Wartości z `.env` nie są poprawnie wczytywane\n\n"
                "**Sprawdź plik `.env` - powinien wyglądać tak (bez cudzysłowów, bez spacji na końcu):**\n"
                "```\n"
                "QDRANT_URL=https://07ba8a0d-dc56-4964-9850-6fe9b8110e1e.eu-central-1-0.aws.cloud.qdrant.io\n"
                "QDRANT_API_KEY=TWÓJ_KLUCZ\n"
                "```\n\n"
                f"**Wczytany QDRANT_URL:** {url_from_env or '(brak)'}\n"
                f"**Wczytany QDRANT_API_KEY (początek):** {key_preview}\n\n"
                f"**Szczegóły błędu:** {e}\n\n"
                "**Rozwiązanie:**\n"
                "1. Sprawdź plik `.env` - usuń wszystkie spacje i nowe linie z wartości\n"
                "2. Zrestartuj Streamlit (Ctrl+C, potem ponownie `streamlit run notatki_app.py`)\n"
                "3. Wymuś odświeżenie cache: Menu → Settings → Clear cache")
        elif error_status == 404 or '404' in str(e) or 'not found' in error_str:
            st.error(
                "🔍 **Nie mogę połączyć się z Qdrant (404 Not Found)**\n\n"
                "Sprawdź poprawność QDRANT_URL.\n\n"
                "**Dla Qdrant Cloud użyj URL bez portu:**\n"
                "`https://xxxx-xxxx.cloud.qdrant.io`\n\n"
                f"**Aktualny QDRANT_URL:** {env.get('QDRANT_URL', '(brak)')}\n\n"
                f"**Szczegóły:** {e}")
        else:
            st.error(
                f"⚠️ **Błąd połączenia z Qdrant**\n\n"
                f"**Status:** {error_status or 'nieznany'}\n\n"
                f"**Aktualny QDRANT_URL:** {env.get('QDRANT_URL', '(brak)')}\n\n"
                f"**Szczegóły:** {e}")
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

def convert_audio_to_mp3(audio_bytes, input_format="m4a"):
    """Konwertuje plik audio do formatu MP3"""
    try:
        # Wczytaj audio z BytesIO
        audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format=input_format)
        # Eksportuj do MP3
        output = BytesIO()
        audio_segment.export(output, format="mp3")
        output.seek(0)
        return output.getvalue()
    except Exception as e:
        st.warning(f"⚠️ Nie udało się przekonwertować pliku do MP3: {e}")
        st.info("💡 Spróbuję użyć oryginalnego pliku...")
        return None

def transcribe_audio_with_timestamps(audio_bytes, filename="audio.mp3", language=None, prompt=None, temperature=0.0):
    """Transkrypcja audio ze znacznikami czasu z opcjami poprawy precyzji
    
    Args:
        audio_bytes: Bajty pliku audio
        filename: Nazwa pliku
        language: Kod języka (np. 'pl', 'en') - opcjonalny, pomaga poprawić precyzję
        prompt: Podpowiedź/kontekst dla modelu (np. słowa kluczowe, nazwy własne) - opcjonalny
        temperature: Temperatura (0.0-1.0), niższa = bardziej deterministyczne, domyślnie 0.0
    """
    openai_client = get_openai_client()
    
    # Zapisz oryginalne dane na wypadek potrzeby użycia później
    original_audio_bytes = audio_bytes
    original_filename = filename
    
    # Upewnij się, że nazwa pliku ma prawidłowe rozszerzenie
    filename_lower = filename.lower()
    valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mpeg', '.mpga', '.oga']
    
    # Sprawdź format pliku
    file_ext = None
    for ext in valid_extensions:
        if filename_lower.endswith(ext):
            file_ext = ext[1:]  # Usuń kropkę
            break
    
    # Użyj przekonwertowanego pliku jeśli jest dostępny, w przeciwnym razie użyj oryginalnego
    if st.session_state.get("file_audio_converted_mp3") and st.session_state.get("use_converted_for_transcription", False):
        audio_bytes = st.session_state["file_audio_converted_mp3"]
        filename = filename.rsplit('.', 1)[0] + '.mp3' if '.' in filename else filename + '.mp3'
    
    audio_file = BytesIO(audio_bytes)
    
    # Jeśli nazwa nie ma rozszerzenia, dodaj .mp3
    if not any(filename_lower.endswith(ext) for ext in valid_extensions):
        filename = f"{filename}.mp3" if '.' not in filename else filename.rsplit('.', 1)[0] + '.mp3'
    
    audio_file.name = filename
    audio_file.seek(0)
    
    # Przygotuj parametry API
    api_params = {
        "file": audio_file,
        "model": AUDIO_TRANSCRIBE_MODEL,
        "response_format": "verbose_json",
        "timestamp_granularities": ["segment"]
    }
    
    # Dodaj opcjonalne parametry poprawiające precyzję
    if language:
        api_params["language"] = language
    if prompt:
        api_params["prompt"] = prompt
    if temperature is not None:
        api_params["temperature"] = max(0.0, min(1.0, temperature))  # Ogranicz do zakresu 0.0-1.0
    
    try:
        transcript = openai_client.audio.transcriptions.create(**api_params)
    except Exception as e:
        error_msg = str(e)
        # Jeśli użyliśmy konwersji i to nie zadziałało, spróbuj z oryginalnym plikiem
        if st.session_state.get("use_converted_for_transcription", False) and st.session_state.get("file_audio_converted_mp3"):
            st.warning("⚠️ Transkrypcja przekonwertowanego pliku nie powiodła się. Próbuję z oryginalnym plikiem...")
            try:
                original_file = BytesIO(original_audio_bytes)
                original_file.name = original_filename
                original_file.seek(0)
                # Użyj tych samych parametrów dla oryginalnego pliku
                original_api_params = {
                    "file": original_file,
                    "model": AUDIO_TRANSCRIBE_MODEL,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                if language:
                    original_api_params["language"] = language
                if prompt:
                    original_api_params["prompt"] = prompt
                if temperature is not None:
                    original_api_params["temperature"] = max(0.0, min(1.0, temperature))
                
                transcript = openai_client.audio.transcriptions.create(**original_api_params)
                st.success("✅ Transkrypcja z oryginalnym plikiem powiodła się!")
            except Exception as e2:
                st.error(f"❌ Błąd transkrypcji audio: {error_msg}")
                st.error(f"❌ Błąd również z oryginalnym plikiem: {e2}")
                st.info(f"💡 Próbowano użyć nazwy pliku: {filename} (przekonwertowany) i {original_filename} (oryginalny)")
                return None
        else:
            st.error(f"❌ Błąd transkrypcji audio: {error_msg}")
            st.info(f"💡 Próbowano użyć nazwy pliku: {filename}")
            return None
    
    # Parsowanie odpowiedzi
    if isinstance(transcript, dict):
        return transcript
    else:
        # Konwersja obiektu do słownika
        result = {
            "text": getattr(transcript, "text", ""),
            "language": getattr(transcript, "language", ""),
            "duration": getattr(transcript, "duration", 0),
            "segments": []
        }
        if hasattr(transcript, "segments"):
            segments = transcript.segments
            if segments:
                for seg in segments:
                    result["segments"].append({
                        "id": getattr(seg, "id", 0),
                        "start": getattr(seg, "start", 0),
                        "end": getattr(seg, "end", 0),
                        "text": getattr(seg, "text", "")
                    })
        return result

def load_audio_from_url(url):
    """Wczytuje plik audio z URL i zwraca (BytesIO, filename)"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Określ format na podstawie URL lub Content-Type
        filename = "audio.mp3"  # domyślny format
        content_type = response.headers.get('Content-Type', '').lower()
        
        # Mapowanie Content-Type na rozszerzenia (zgodne z OpenAI API)
        content_type_map = {
            'audio/mpeg': 'audio.mp3',
            'audio/mp3': 'audio.mp3',
            'audio/x-mpeg': 'audio.mp3',
            'audio/wav': 'audio.wav',
            'audio/wave': 'audio.wav',
            'audio/x-wav': 'audio.wav',
            'audio/mp4': 'audio.m4a',
            'audio/x-m4a': 'audio.m4a',
            'audio/m4a': 'audio.m4a',
            'audio/ogg': 'audio.ogg',
            'audio/oga': 'audio.oga',
            'audio/vorbis': 'audio.ogg',
            'audio/flac': 'audio.flac',
            'audio/x-flac': 'audio.flac',
            'audio/webm': 'audio.webm',
            'audio/mpga': 'audio.mpga',
        }
        
        if content_type in content_type_map:
            filename = content_type_map[content_type]
        else:
            # Spróbuj określić z URL
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            if path.endswith(('.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mpeg', '.mpga', '.oga')):
                ext = path[path.rfind('.'):]
                filename = f"audio{ext}"
        
        return BytesIO(response.content), filename
    except Exception as e:
        st.error(f"Błąd podczas pobierania pliku z URL: {e}")
        return None, None

def format_timestamp(seconds):
    """Formatuje czas w sekundach na format HH:MM:SS lub MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def register_polish_fonts():
    """Rejestruje fonty obsługujące polskie znaki"""
    try:
        from reportlab.lib.fonts import addMapping
        
        # Sprawdź dostępne fonty systemowe obsługujące polskie znaki
        font_paths = []
        
        # Windows
        if os.name == 'nt':
            windows_fonts_dir = 'C:/Windows/Fonts'
            font_paths.extend([
                os.path.join(windows_fonts_dir, 'arial.ttf'),
                os.path.join(windows_fonts_dir, 'calibri.ttf'),
                os.path.join(windows_fonts_dir, 'times.ttf'),
            ])
        # Linux
        else:
            font_paths.extend([
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/TTF/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
            ])
        
        # Spróbuj zarejestrować pierwszy dostępny font
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('PolishFont', font_path))
                    addMapping('PolishFont', 0, 0, 'PolishFont')
                    return True
                except Exception:
                    continue
        
        return False
    except Exception:
        return False

def generate_pdf(transcript_data, output_filename="transkrypcja.pdf"):
    """Generuje PDF z transkrypcji ze znacznikami czasu"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    
    # Zarejestruj fonty obsługujące polskie znaki
    font_registered = register_polish_fonts()
    font_name = 'PolishFont' if font_registered else 'Helvetica'
    
    # Style z fontem obsługując polskie znaki - zmniejszone rozmiary
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=14,
        textColor=HexColor('#1567ea'),
        spaceAfter=8,
        alignment=TA_LEFT,
        encoding='utf-8'
    )
    timestamp_style = ParagraphStyle(
        'Timestamp',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=8,
        textColor=HexColor('#666666'),
        spaceAfter=0,
        alignment=TA_LEFT,
        encoding='utf-8'
    )
    text_style = ParagraphStyle(
        'Text',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        spaceAfter=6,
        alignment=TA_LEFT,
        encoding='utf-8',
        leading=11  # Odstęp między liniami w paragrafie
    )
    
    # Styl dla normalnego tekstu również
    normal_style = ParagraphStyle(
        'NormalPolish',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        encoding='utf-8',
        spaceAfter=4
    )
    
    # Styl dla linii z timestampem i tekstem w jednej linii
    inline_style = ParagraphStyle(
        'InlineText',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=9,
        spaceAfter=6,
        alignment=TA_LEFT,
        encoding='utf-8',
        leading=11
    )
    
    # Zawartość PDF
    story = []
    
    # Tytuł
    story.append(Paragraph("Transkrypcja audio", title_style))
    story.append(Spacer(1, 0.1*inch))
    
    # Informacje o transkrypcji
    if transcript_data:
        if isinstance(transcript_data, dict):
            language = transcript_data.get("language", "nieznany")
            duration = transcript_data.get("duration", 0)
            
            # Funkcja pomocnicza do escapowania HTML i zachowania polskich znaków
            def escape_html(text):
                """Escapuje HTML, zachowując polskie znaki"""
                if not text:
                    return ""
                text = str(text)
                text = text.replace('&', '&amp;')
                text = text.replace('<', '&lt;')
                text = text.replace('>', '&gt;')
                return text
            
            story.append(Paragraph(f"<b>Język:</b> {escape_html(language)}", normal_style))
            story.append(Paragraph(f"<b>Czas trwania:</b> {format_timestamp(duration)}", normal_style))
            story.append(Paragraph(f"<b>Data transkrypcji:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
            story.append(Spacer(1, 0.15*inch))
            
            # Segmenty z timestampami - timestamp i tekst w jednej linii
            segments = transcript_data.get("segments", [])
            if segments:
                for segment in segments:
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "")
                    
                    # Timestamp i tekst w jednej linii
                    timestamp_text = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                    escaped_text = escape_html(text)
                    # Połącz timestamp i tekst w jednej linii
                    combined_text = f"<b>{timestamp_text}</b> {escaped_text}"
                    story.append(Paragraph(combined_text, inline_style))
            else:
                # Jeśli brak segmentów, użyj pełnego tekstu
                full_text = transcript_data.get("text", "")
                if full_text:
                    escaped_text = escape_html(full_text)
                    story.append(Paragraph(escaped_text, text_style))
    
    # Budowanie PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def format_transcript_as_text(transcript_data):
    """Formatuje transkrypcję jako tekst do zapisania jako notatka"""
    lines = []
    
    if transcript_data and isinstance(transcript_data, dict):
        language = transcript_data.get("language", "nieznany")
        duration = transcript_data.get("duration", 0)
        
        lines.append(f"Transkrypcja audio")
        lines.append(f"Język: {language}")
        lines.append(f"Czas trwania: {format_timestamp(duration)}")
        lines.append(f"Data transkrypcji: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Segmenty z timestampami
        segments = transcript_data.get("segments", [])
        if segments:
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                
                # Timestamp i tekst w jednej linii
                timestamp_text = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                lines.append(f"{timestamp_text} {text}")
        else:
            # Jeśli brak segmentów, użyj pełnego tekstu
            full_text = transcript_data.get("text", "")
            if full_text:
                lines.append(full_text)
    
    return "\n".join(lines)

def generate_txt(transcript_data, output_filename="transkrypcja.txt"):
    """Generuje plik TXT z transkrypcji ze znacznikami czasu"""
    buffer = BytesIO()
    
    lines = []
    lines.append("=" * 60)
    lines.append("TRANSKRYPCJA AUDIO")
    lines.append("=" * 60)
    lines.append("")
    
    if transcript_data and isinstance(transcript_data, dict):
        language = transcript_data.get("language", "nieznany")
        duration = transcript_data.get("duration", 0)
        
        lines.append(f"Język: {language}")
        lines.append(f"Czas trwania: {format_timestamp(duration)}")
        lines.append(f"Data transkrypcji: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        lines.append("-" * 60)
        lines.append("")
        
        # Segmenty z timestampami
        segments = transcript_data.get("segments", [])
        if segments:
            for segment in segments:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "")
                
                # Timestamp i tekst w jednej linii
                timestamp_text = f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"
                lines.append(f"{timestamp_text} {text}")
        else:
            # Jeśli brak segmentów, użyj pełnego tekstu
            full_text = transcript_data.get("text", "")
            if full_text:
                lines.append(full_text)
    
    # Konwersja do bytes z kodowaniem UTF-8
    text_content = "\n".join(lines)
    buffer.write(text_content.encode('utf-8'))
    buffer.seek(0)
    return buffer

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

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

if "file_transcript_data" not in st.session_state:
    st.session_state["file_transcript_data"] = None

if "file_audio_bytes" not in st.session_state:
    st.session_state["file_audio_bytes"] = None

if "file_audio_filename" not in st.session_state:
    st.session_state["file_audio_filename"] = "audio.mp3"

if "file_audio_converted_mp3" not in st.session_state:
    st.session_state["file_audio_converted_mp3"] = None

if "file_audio_original_format" not in st.session_state:
    st.session_state["file_audio_original_format"] = None

if "file_audio_converted_mp3" not in st.session_state:
    st.session_state["file_audio_converted_mp3"] = None

if "file_audio_original_format" not in st.session_state:
    st.session_state["file_audio_original_format"] = None

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
Wyszukiwanie w zapisanych działa semantycznie z wykorzystaniem modelu AI.
""", unsafe_allow_html=True)

db_configured = bool(env.get("QDRANT_URL") and env.get("QDRANT_API_KEY"))

if db_configured:
    assure_db_collection_exists()
else:
    st.warning("Konfiguracja Qdrant nie ustawiona. Aby zapisywać notatki, ustaw QDRANT_URL i QDRANT_API_KEY w .env lub Secrets.")

selected = option_menu(None, ["Dodaj notatkę", "Wyszukaj notatkę", "Transkrypcja z pliku"],
    icons=['record', 'search', 'file-audio'], 
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
elif selected == "Transkrypcja z pliku":
    st.markdown("### 📁 Wczytaj plik audio z dysku lub URL")
    
    # Wybór źródła audio
    source_type = st.radio(
        "Wybierz źródło audio:",
        ["Plik z dysku", "URL"],
        horizontal=True
    )
    
    audio_bytes = None
    audio_filename = "audio.mp3"
    
    if source_type == "Plik z dysku":
        uploaded_file = st.file_uploader(
            "Wybierz plik audio",
            type=['mp3', 'wav', 'm4a', 'ogg', 'flac', 'webm', 'mp4', 'mpeg', 'mpga', 'oga'],
            help="Obsługiwane formaty: MP3, WAV, M4A, OGG, FLAC, WEBM, MP4, MPEG, MPGA, OGA"
        )
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            
            # Użyj dokładnie nazwy pliku, jeśli ma rozszerzenie audio
            if uploaded_file.name:
                audio_filename = uploaded_file.name
                # Sprawdź czy nazwa ma prawidłowe rozszerzenie audio
                audio_filename_lower = audio_filename.lower()
                valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mpeg', '.mpga', '.oga']
                has_valid_extension = any(audio_filename_lower.endswith(ext) for ext in valid_extensions)
                
                if not has_valid_extension:
                    # Określ rozszerzenie na podstawie typu MIME
                    mime_to_ext = {
                        'audio/mpeg': '.mp3',
                        'audio/mp3': '.mp3',
                        'audio/x-mpeg': '.mp3',
                        'audio/wav': '.wav',
                        'audio/wave': '.wav',
                        'audio/x-wav': '.wav',
                        'audio/mp4': '.m4a',
                        'audio/x-m4a': '.m4a',
                        'audio/m4a': '.m4a',
                        'audio/ogg': '.ogg',
                        'audio/oga': '.oga',
                        'audio/vorbis': '.ogg',
                        'audio/flac': '.flac',
                        'audio/x-flac': '.flac',
                        'audio/webm': '.webm',
                        'audio/mpeg': '.mpeg',
                        'audio/mpga': '.mpga',
                    }
                    ext = mime_to_ext.get(uploaded_file.type, '.m4a')
                    audio_filename = f"{audio_filename}{ext}"
            else:
                # Określ na podstawie typu MIME
                mime_to_ext = {
                    'audio/mpeg': '.mp3',
                    'audio/mp3': '.mp3',
                    'audio/x-mpeg': '.mp3',
                    'audio/wav': '.wav',
                    'audio/wave': '.wav',
                    'audio/x-wav': '.wav',
                    'audio/mp4': '.m4a',
                    'audio/x-m4a': '.m4a',
                    'audio/m4a': '.m4a',
                    'audio/ogg': '.ogg',
                    'audio/oga': '.oga',
                    'audio/vorbis': '.ogg',
                    'audio/flac': '.flac',
                    'audio/x-flac': '.flac',
                    'audio/webm': '.webm',
                    'audio/mpeg': '.mpeg',
                    'audio/mpga': '.mpga',
                }
                ext = mime_to_ext.get(uploaded_file.type, '.m4a')
                audio_filename = f"audio{ext}"
            
            # Debug: wyświetl informacje o pliku
            st.info(f"📁 Wczytano plik: {audio_filename} (typ MIME: {uploaded_file.type or 'nieznany'})")
            
            # Wyczyść poprzednie konwersje TYLKO jeśli wczytujemy nowy plik (nie po konwersji)
            # Sprawdź czy to nowy plik porównując nazwę pliku
            current_filename = st.session_state.get("file_audio_filename")
            if current_filename != audio_filename:
                st.session_state["file_audio_converted_mp3"] = None
                st.session_state["file_audio_original_format"] = None
                st.session_state["use_converted_for_transcription"] = False
            
            st.session_state["file_audio_bytes"] = audio_bytes
            st.session_state["file_audio_filename"] = audio_filename
            st.audio(audio_bytes, format=uploaded_file.type or "audio/mp3")
    
    elif source_type == "URL":
        url = st.text_input("Wprowadź URL pliku audio", placeholder="https://example.com/audio.mp3")
        if url:
            with st.spinner("Pobieranie pliku audio..."):
                result = load_audio_from_url(url)
                if result and result[0] is not None:
                    audio_bytes, audio_filename = result
                    
                    # Wyczyść poprzednie konwersje TYLKO jeśli wczytujemy nowy plik (nie po konwersji)
                    current_filename = st.session_state.get("file_audio_filename")
                    if current_filename != audio_filename:
                        st.session_state["file_audio_converted_mp3"] = None
                        st.session_state["file_audio_original_format"] = None
                        st.session_state["use_converted_for_transcription"] = False
                    
                    st.session_state["file_audio_bytes"] = audio_bytes.getvalue()
                    st.session_state["file_audio_filename"] = audio_filename
                    st.audio(audio_bytes.getvalue(), format="audio/mp3")
                    audio_bytes = audio_bytes.getvalue()
    
    # Inicjalizacja nazwy pliku w sesji
    if "file_audio_filename" not in st.session_state:
        st.session_state["file_audio_filename"] = "audio.mp3"
    
    # Transkrypcja ze znacznikami czasu
    if audio_bytes or st.session_state.get("file_audio_bytes"):
        audio_to_transcribe = audio_bytes if audio_bytes else st.session_state["file_audio_bytes"]
        filename_to_use = audio_filename if audio_bytes else st.session_state.get("file_audio_filename", "audio.mp3")
        
        # Wyświetl informację o pliku
        st.info(f"📁 Plik: {filename_to_use}")
        
        # Sprawdź format pliku i możliwość konwersji
        filename_lower = filename_to_use.lower()
        file_ext = None
        valid_extensions = ['.mp3', '.wav', '.m4a', '.ogg', '.flac', '.webm', '.mp4', '.mpeg', '.mpga', '.oga']
        for ext in valid_extensions:
            if filename_lower.endswith(ext):
                file_ext = ext[1:]  # Usuń kropkę
                break
        
        # Przycisk konwersji do MP3 (jeśli plik nie jest już MP3)
        if file_ext and file_ext not in ['mp3']:
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🔄 Konwertuj do MP3", help="Konwertuj plik do formatu MP3 przed transkrypcją. Może poprawić kompatybilność z API."):
                    with st.spinner("Konwersja w toku..."):
                        try:
                            converted_audio = convert_audio_to_mp3(audio_to_transcribe, input_format=file_ext)
                            if converted_audio:
                                st.session_state["file_audio_converted_mp3"] = converted_audio
                                st.session_state["file_audio_original_format"] = file_ext
                                st.session_state["use_converted_for_transcription"] = True
                                st.toast(f"✅ Plik przekonwertowany z {file_ext.upper()} do MP3!", icon="✅")
                                st.rerun()
                            else:
                                st.error("❌ Konwersja nie powiodła się")
                        except Exception as e:
                            st.error(f"❌ Błąd konwersji: {e}")
        
        # Opcja pobrania/zapisu przekonwertowanego pliku MP3 (jeśli został przekonwertowany)
        # Wyświetlane poza warunkiem formatu, aby były zawsze widoczne po konwersji
        converted_audio_exists = st.session_state.get("file_audio_converted_mp3") is not None
        original_format_exists = st.session_state.get("file_audio_original_format") is not None
        
        if converted_audio_exists and original_format_exists:
            original_format = st.session_state["file_audio_original_format"].upper()
            # Użyj nazwy pliku z session_state, aby była zawsze dostępna po przeładowaniu
            base_filename = st.session_state.get("file_audio_filename", filename_to_use)
            converted_filename = base_filename.rsplit('.', 1)[0] + '.mp3' if '.' in base_filename else base_filename + '.mp3'
            
            st.success(f"✅ Plik przekonwertowany z {original_format} do MP3")
            
            # Przycisk zapisu/pobierania
            st.download_button(
                label=f"💾 Zapisz MP3 na dysk",
                data=st.session_state["file_audio_converted_mp3"],
                file_name=converted_filename,
                mime="audio/mpeg",
                help=f"Zapisz przekonwertowany plik MP3 na dysk (pobieranie pliku)",
                type="primary",
                key=f"download_converted_mp3_{hash(converted_filename)}"
            )
            
            # Opcja wyboru czy używać przekonwertowanego pliku do transkrypcji
            use_converted = st.checkbox(
                "Użyj przekonwertowanego pliku MP3 do transkrypcji",
                value=st.session_state.get("use_converted_for_transcription", False),
                help="Zaznacz, aby użyć przekonwertowanego pliku MP3 zamiast oryginalnego podczas transkrypcji"
            )
            st.session_state["use_converted_for_transcription"] = use_converted
        
        # Opcje poprawy precyzji transkrypcji
        with st.expander("⚙️ Opcje poprawy precyzji transkrypcji (opcjonalne)"):
            st.markdown("**Te opcje mogą poprawić jakość transkrypcji:**")
            
            # Wybór języka
            language_option = st.selectbox(
                "Język (opcjonalnie - pomaga poprawić precyzję)",
                ["Automatyczny", "Polski (pl)", "Angielski (en)", "Niemiecki (de)", "Francuski (fr)", "Hiszpański (es)", "Włoski (it)", "Rosyjski (ru)"],
                help="Jeśli znasz język nagrania, wybierz go. To pomoże modelowi lepiej rozpoznać słowa."
            )
            
            language_code = None
            if language_option != "Automatyczny":
                # Wyciągnij kod języka z opcji (np. "Polski (pl)" -> "pl")
                language_code = language_option.split("(")[1].split(")")[0] if "(" in language_option else None
            
            # Prompt/podpowiedź
            prompt_text = st.text_area(
                "Podpowiedź/kontekst (opcjonalnie)",
                placeholder="Wpisz słowa kluczowe, nazwy własne, terminy techniczne lub kontekst nagrania, które mogą pojawić się w transkrypcji. To pomoże modelowi lepiej rozpoznać specjalistyczne słowa.",
                help="Np. nazwy własne, terminy techniczne, słowa kluczowe związane z tematem nagrania"
            )
            
            # Temperatura
            temperature_value = st.slider(
                "Temperatura (0.0 = bardziej deterministyczne, 1.0 = bardziej kreatywne)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Niższa temperatura (0.0) daje bardziej precyzyjne i deterministyczne wyniki. Wyższa temperatura może być bardziej kreatywna, ale mniej precyzyjna."
            )
        
        if st.button("🖋️ Transkrybuj ze znacznikami czasu", type="primary"):
            with st.spinner("Transkrypcja w toku... To może chwilę potrwać."):
                transcript_data = transcribe_audio_with_timestamps(
                    audio_to_transcribe, 
                    filename_to_use,
                    language=language_code if language_code else None,
                    prompt=prompt_text if prompt_text else None,
                    temperature=temperature_value
                )
                if transcript_data:
                    st.session_state["file_transcript_data"] = transcript_data
                    st.success("Transkrypcja zakończona!")
        
        # Wyświetlanie transkrypcji
        if st.session_state["file_transcript_data"]:
            transcript_data = st.session_state["file_transcript_data"]
            
            st.markdown("### 📝 Transkrypcja ze znacznikami czasu")
            
            # Informacje o transkrypcji
            col1, col2, col3 = st.columns(3)
            with col1:
                language = transcript_data.get("language", "nieznany")
                st.metric("Język", language.upper() if language else "Nieznany")
            with col2:
                duration = transcript_data.get("duration", 0)
                st.metric("Czas trwania", format_timestamp(duration))
            with col3:
                segments_count = len(transcript_data.get("segments", []))
                st.metric("Liczba segmentów", segments_count)
            
            st.divider()
            
            # Wyświetlanie segmentów
            segments = transcript_data.get("segments", [])
            if segments:
                st.markdown("#### Segmenty transkrypcji:")
                for i, segment in enumerate(segments, 1):
                    start_time = segment.get("start", 0)
                    end_time = segment.get("end", 0)
                    text = segment.get("text", "")
                    
                    with st.expander(f"[{format_timestamp(start_time)} - {format_timestamp(end_time)}]"):
                        st.write(text)
            else:
                # Jeśli brak segmentów, pokaż pełny tekst
                full_text = transcript_data.get("text", "")
                if full_text:
                    st.text_area("Pełna transkrypcja", full_text, height=200)
            
            st.divider()
            
            # Opcja zapisania transkrypcji jako notatki
            if db_configured:
                st.markdown("### 💾 Zapisz jako notatkę")
                if st.button("📝 Zapisz transkrypcję jako notatkę", type="primary", help="Zapisze całą transkrypcję jako notatkę w bazie danych"):
                    try:
                        transcript_text = format_transcript_as_text(transcript_data)
                        if transcript_text:
                            add_note_to_db(note_text=transcript_text)
                            st.toast("✅ Transkrypcja zapisana jako notatka!", icon="💾")
                        else:
                            st.error("❌ Nie można zapisać pustej transkrypcji")
                    except Exception as e:
                        st.error(f"❌ Błąd podczas zapisywania notatki: {e}")
            else:
                st.info("ℹ️ Aby zapisać transkrypcję jako notatkę, skonfiguruj QDRANT_URL i QDRANT_API_KEY w .env lub Secrets.")
            
            st.divider()
            
            # Generowanie pliku
            st.markdown("### 📄 Generuj plik")
            
            # Wybór formatu
            file_format = st.radio(
                "Wybierz format pliku:",
                ["PDF", "TXT"],
                horizontal=True
            )
            
            # Nazwa pliku
            base_filename = f"transkrypcja_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if file_format == "PDF":
                default_filename = f"{base_filename}.pdf"
                file_extension = ".pdf"
                mime_type = "application/pdf"
            else:
                default_filename = f"{base_filename}.txt"
                file_extension = ".txt"
                mime_type = "text/plain"
            
            filename_input = st.text_input(
                f"Nazwa pliku {file_format}",
                value=default_filename
            )
            
            if st.button(f"📥 Generuj i pobierz {file_format}"):
                # Upewnij się, że plik ma poprawne rozszerzenie
                if filename_input:
                    if not filename_input.endswith(file_extension):
                        filename_input = filename_input.rsplit('.', 1)[0] + file_extension
                else:
                    filename_input = default_filename
                
                if file_format == "PDF":
                    file_buffer = generate_pdf(transcript_data, filename_input)
                    label = "⬇️ Pobierz PDF"
                else:
                    file_buffer = generate_txt(transcript_data, filename_input)
                    label = "⬇️ Pobierz TXT"
                
                st.download_button(
                    label=label,
                    data=file_buffer,
                    file_name=filename_input,
                    mime=mime_type
                )
