from fastapi import FastAPI, HTTPException

from app.auth import validate_token
from app.schemas import LoginRequest

app = FastAPI()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/login")
def login(request: LoginRequest):
    if not validate_token(request.token):
        raise HTTPException(status_code=401, detail="Invalid token")
    return {"message": "Login successful"}
