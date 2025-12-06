@echo off
echo [WARLORD] Uruchamianie procedury Tissaia V14...
if not exist .env ( echo [ERROR] Brak .env! & pause & exit )
docker compose up --build
pause