@echo off
echo [WARLORD] Tissaia V23 (Anti-Strip Protocol)...
if not exist .env ( echo [ERROR] Brak .env! & pause & exit )
docker compose up --build
pause