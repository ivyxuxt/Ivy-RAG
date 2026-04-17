import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.models import HealthResponse, ReadyResponse

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)

# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directories exist
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    os.makedirs(settings.raw_dir, exist_ok=True)

    # Load index into memory on startup
    from app.storage.store import store
    store.load()

    yield


app = FastAPI(
    title="Ivy RAG",
    description="PDF-based Retrieval-Augmented Generation API",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Optional API key guard. Enabled only when API_KEY is set in .env.
    Clients must send: X-Api-Key: <value>
    Health endpoints are exempt so Docker healthchecks work without a key.
    """
    if settings.API_KEY:
        exempt = request.url.path in ("/health", "/ready")
        if not exempt:
            provided = request.headers.get("X-Api-Key", "")
            if provided != settings.API_KEY:
                return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
from app.ingestion.pipeline import router as ingest_router
from app.generation.llm import router as query_router

app.include_router(ingest_router)
app.include_router(query_router)


# ── Health endpoints ───────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    return HealthResponse(ok=True)


@app.get("/ready", response_model=ReadyResponse, tags=["health"])
def ready():
    from app.storage.store import store
    return ReadyResponse(ok=True, chunks_loaded=len(store.chunks))
