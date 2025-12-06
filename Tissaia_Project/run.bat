@echo off
:: Tissaia Runner V4 - Dev Mode (Smart Build)
:: Agent: The Warlord

:: 1. SPRAWDZENIE KLUCZA
IF "%GOOGLE_API_KEY%"=="" (
    echo [ERROR] Brak zmiennej GOOGLE_API_KEY w systemie Windows!
    echo Ustaw ja komenda: setx GOOGLE_API_KEY "twoj-klucz"
    pause
    exit /b
)

:: 2. SMART BUILD (Wymuszenie weryfikacji zmian w kodzie/bibliotekach)
echo [INFO] Weryfikacja spojnosci obrazu (Rebuilding if needed)...
docker build -t tissaia-app .

:: Sprawdzenie bledu budowania
IF %ERRORLEVEL% NEQ 0 (
    echo [CRITICAL] Blad budowania Dockerfile! Sprawdz logi.
    pause
    exit /b
)
echo [INFO] Obraz gotowy i uzbrojony.

:: 3. URUCHOMIENIE
echo [ACTION] Uruchamianie kontenera...
docker run --rm -v "%cd%:/app" -e GOOGLE_API_KEY="%GOOGLE_API_KEY%" tissaia-app

echo [DONE] Misja zakonczona.
pause