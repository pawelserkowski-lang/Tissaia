@echo off
setlocal enabledelayedexpansion

echo [ARCHITECT] Inicjalizacja procedury Dockerowej...

:: 1. TWORZENIE PLIKÓW KONFIGURACYJNYCH (Omijamy Pythona)

echo [1/4] Generowanie requirements.txt...
(
echo opencv-python-headless
echo numpy
) > requirements.txt

echo [2/4] Generowanie Dockerfile...
(
echo FROM python:3.11-slim
echo WORKDIR /app
echo ENV PYTHONDONTWRITEBYTECODE=1
echo ENV PYTHONUNBUFFERED=1
echo RUN apt-get update ^&^& apt-get install -y --no-install-recommends libgl1 libglib2.0-0 ^&^& rm -rf /var/lib/apt/lists/*
echo COPY requirements.txt .
echo RUN pip install --no-cache-dir -r requirements.txt
echo COPY . .
echo CMD ["python", "verify_tissaia_v14.py"]
) > Dockerfile

echo [3/4] Generowanie docker-compose.yml...
(
echo services:
echo   tissaia-verify:
echo     build: .
echo     image: tissaia-v14-total-war
echo     volumes:
echo       - .:/app
echo     command: python verify_tissaia_v14.py
) > docker-compose.yml

:: 2. WERYFIKACJA PLIKÓW

if not exist "verify_tissaia_v14.py" (
    echo [ERROR] Brak pliku verify_tissaia_v14.py!
    echo Zapisz kod z prawej strony ekranu do folderu projektu.
    pause
    exit /b
)

if not exist "zdjecia.zip" (
    echo [WARNING] Nie widze zdjecia.zip. Skrypt sprobuje je utworzyc z plikow JPG w folderze.
)

:: 3. URUCHOMIENIE

echo [4/4] Uruchamianie PROTOKOLU TOTAL WAR w kontenerze...
echo -----------------------------------------------------
docker compose up --build

echo -----------------------------------------------------
echo [WARLORD] Misja zakonczona. Jesli widzisz powyzszy raport, wygralismy.
pause