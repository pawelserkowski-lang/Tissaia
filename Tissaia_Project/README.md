# Tissaia Project - Gemini Restoration API

A backend service for high-fidelity photo restoration using Google Gemini 3 Vision.

## Architecture

This project uses FastAPI as the web server and interacts with Google's Generative AI models.
It includes a job management system to handle image processing asynchronously.

### Key Components

* **API Server (`api_server.py`)**: FastAPI application exposing endpoints for upload and processing.
* **Pipeline (`src/pipeline.py`)**: Manages jobs (`JobManager`) and executes the restoration workflow.
* **AI Core (`src/ai_core.py`)**: Handles communication with Gemini API.
* **Graphics (`src/graphics.py`)**: OpenCV/PIL based image pre/post-processing.

## Setup & Running

### Prerequisites

* Docker & Docker Compose
* Google Cloud API Key (with Gemini Vision access)

### Deployment

1.  **Configure Environment**:
    Ensure `GOOGLE_API_KEY` is set in your system environment variables OR paste it into `.env`.

2.  **Run with Docker**:
    ```bash
    cd Tissaia_Project
    docker compose up --build -d
    ```

3.  **Access API**:
    * Swagger UI: `http://localhost:8000/docs`
    * System Status: `http://localhost:8000/system/status`
