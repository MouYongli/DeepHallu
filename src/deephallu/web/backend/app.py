from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from .api.datasets import router as datasets_router
from .core.config import config

app = FastAPI(
    title="DeepHallu Web Interface",
    description="On the Analysis of Hallucination in Vision Language Models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(datasets_router)

@app.get("/")
async def root():
    return {"message": "DeepHallu Web Interface", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("app:app", host=config.server.host, port=config.server.port, reload=config.server.reload)