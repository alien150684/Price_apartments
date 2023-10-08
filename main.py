from fastapi import FastAPI
from uvicorn import run

from api.api import app  # Импортируем экземпляр приложения из api.py

if __name__ == "__main__":
    run(app, host="127.0.0.1", port=8000) # чтобы нельзя было запустить при импорте