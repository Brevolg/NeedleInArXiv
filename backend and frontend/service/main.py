from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from search.config import get_settings
from search.engine import SearchEngine
from service.schemas import ResultItem, SearchRequest, SearchResponseModel


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.engine = await run_in_threadpool(SearchEngine, get_settings())
        app.state.startup_error = None
    except Exception as exc:
        app.state.engine = None
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="Semantic Search",
    version="1.0.0",
    description="Dense, BM25, and RRF hybrid retrieval",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def require_engine(request: Request) -> SearchEngine:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"Search engine is not ready: {request.app.state.startup_error}",
        )
    return engine


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    settings = get_settings()
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"default_mode": settings.default_mode},
    )


@app.get("/health")
async def health(request: Request):
    engine = request.app.state.engine
    if engine is None:
        return {"status": "unavailable", "detail": request.app.state.startup_error}
    return await run_in_threadpool(engine.health)


@app.get("/api/config")
async def config(request: Request):
    engine = require_engine(request)
    return {
        "sources": engine.repository.sources,
        "modes": [
            {"value": "dense_v1", "label": "Iteration 1 — dense exact"},
            {"value": "dense_v2", "label": "Iteration 2 — dense HNSW"},
            {"value": "bm25", "label": "BM25"},
            {"value": "hybrid_v1", "label": "Hybrid v1 — BM25 + dense exact"},
            {"value": "hybrid_v2", "label": "Hybrid v2 — BM25 + dense HNSW"},
        ],
        "default_mode": engine.settings.default_mode,
        "max_top_k": engine.settings.max_top_k,
    }


@app.post("/search", response_model=SearchResponseModel)
@app.post("/api/search", response_model=SearchResponseModel, include_in_schema=False)
async def search(payload: SearchRequest, request: Request) -> SearchResponseModel:
    engine = require_engine(request)
    try:
        response = await run_in_threadpool(
            engine.search,
            payload.query,
            mode=payload.mode,
            top_k=payload.top_k,
            sources=set(payload.sources) if payload.sources else None,
        )
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchResponseModel(
        query=payload.query,
        mode=response.mode,
        latency_ms=response.latency_ms,
        total=len(response.results),
        results=[ResultItem.model_validate(result, from_attributes=True) for result in response.results],
    )

