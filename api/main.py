from fastapi import FastAPI
from api.routes.upload import up_router

app = FastAPI(title="Doc-Explorer — Ingestion API", version="0.1.0")
app.include_router(up_router, prefix="/api/v1")

@app.get("/healthz", tags=["ops"])
def health():
    return {"status": "ok"}