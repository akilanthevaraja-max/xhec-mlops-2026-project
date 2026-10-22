#!/bin/bash

# Start the Prefect server (UI on 4201) in the background, then the FastAPI service (8001)
prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
