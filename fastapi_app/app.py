from fastapi import FastAPI, Query
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Práctica FastAPI - 5 endpoints (2 HF)")

sentiment_pipe = pipeline("sentiment-analysis")  # cargar default model
generator_pipe = pipeline(
    "text-generation",
    model="distilgpt2"
)

# (1) Get simple: Saludar
@app.get("/saludos")
def saludar(name:str):
    return {"mensaje":f"Hola, mi nombre es {name}!"}

# (2) Get simple: Sumar dos enteros
@app.get("/suma")
def suma_dos(a: int, b: int):
    return {"a": a, "b": b, "sum": a + b}

# 3) Get simple: Reverse string
@app.get("/reverse")
def reverse_text(text: str):
    return {"original": text, "reversed": text[::-1]}

# 4) Get con HF pipeline: sentiment analysis
@app.get("/sentiment")
def sentiment(text):
    result = sentiment_pipe(text)
    return {"text": text, "result": result}

# 5) Get con HF pipeline: text translation
@app.get("/generate")
def generate(text: str):
    result = generator_pipe(text, max_length=30, num_return_sequences=1)
    return {"result": result}

