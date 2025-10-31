# notatki_audio

Aplikacja Streamlit do szybkich notatek z transkrypcji audio (OpenAI Whisper) z zapisem i wyszukiwaniem semantycznym w Qdrant.

## Wymagania
- Python 3.10+ (rekomendowany conda env)
- Klucz OpenAI (`OPENAI_API_KEY`)
- Instancja Qdrant (lokalnie lub Qdrant Cloud) + `QDRANT_URL`, `QDRANT_API_KEY`

## Instalacja
```bash
(opcjonalnie) conda create -n notatki_audio python=3.10 -y
conda activate notatki_audio
pip install -r requirements.txt
```

## Konfiguracja
Preferowane lokalnie: plik `.env` w katalogu projektu.

```env
OPENAI_API_KEY=sk-...
QDRANT_URL=https://<twoj-klaster>.cloud.qdrant.io   # Qdrant Cloud – bez portu
QDRANT_API_KEY=<klucz_z_panelu_qdrant>
```

Alternatywnie (np. w Streamlit Cloud): `.streamlit/secrets.toml` lub `~/.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-..."
QDRANT_URL = "https://<twoj-klaster>.cloud.qdrant.io"
QDRANT_API_KEY = "..."
```

Uwagi:
- Dla Qdrant Cloud używaj URL bez portu (np. `https://…cloud.qdrant.io`).
- Klucz API musi mieć uprawnienia do zapisu (nie tylko read-only).

## Uruchomienie
```bash
streamlit run notatki_app.py
```

## Funkcje
- Nagrywanie audio w przeglądarce, transkrypcja (OpenAI Whisper)
- Edycja treści przed zapisem
- Zapis do Qdrant z embeddingami (OpenAI `text-embedding-3-large`)
- Wyszukiwanie semantyczne po notatkach

## Rozwiązywanie problemów
- Brak `secrets.toml` lokalnie: aplikacja korzysta z `.env` – to normalne.
- Logo: jeśli `logo.png` nie istnieje, aplikacja pokaże fallback (emoji) i nie przerwie działania.
- Qdrant 404 (Not Found): sprawdź poprawność `QDRANT_URL` (dla Cloud bez portu).
- Qdrant 403 (Forbidden): sprawdź klucz API i jego uprawnienia; lokalnie upewnij się, że `.env` nie zawiera spacji/nowych linii w wartościach.
- Test połączenia z Qdrant (PowerShell):
  ```powershell
  $Url = "https://<twoj-klaster>.cloud.qdrant.io"
  $Key = "<klucz>"
  Invoke-WebRequest -Uri "$Url/collections" -Headers @{ "api-key" = $Key }
  ```

## Licencja
MIT
