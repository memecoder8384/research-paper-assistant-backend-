from contextlib import asynccontextmanager
import uvicorn
from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.chat import router as chat_router
from app.api.documents import papers_router, router as documents_router
from app.api.health import router as health_router
from app.api.retrieval import router as retrieval_router
from app.api.routes.comparison import router as comparison_router
from app.api.routes.research_gaps import router as gap_router
from app.api.routes.summary import router as summary_router
from app.core.config import settings
from app.core.errors import register_error_handlers
from app.db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown events."""
    # Ensure storage directories and database tables exist on startup
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield


# Initialize FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="Backend API service for the Research Paper Assistant capstone project.",
    version=settings.version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Enable CORS for frontend connectivity (allows React, Vite, Next.js, Streamlit, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register custom exception handlers for clean error responses
register_error_handlers(app)

# Register API routers at root level
app.include_router(health_router)
app.include_router(documents_router)
app.include_router(papers_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
app.include_router(summary_router)
app.include_router(comparison_router)
app.include_router(gap_router)

# Also mount under /api prefix for frontend frameworks configuring VITE_API_BASE_URL=.../api
api_router = APIRouter(prefix="/api")
api_router.include_router(health_router)
api_router.include_router(documents_router)
api_router.include_router(papers_router)
api_router.include_router(retrieval_router)
api_router.include_router(chat_router)
api_router.include_router(summary_router)
api_router.include_router(comparison_router)
api_router.include_router(gap_router)
app.include_router(api_router)


@app.get("/", tags=["Root"], summary="API Root")
async def root():
    """Welcome endpoint pointing to documentation and active services."""
    return {
        "message": f"Welcome to {settings.app_name}",
        "docs": "/docs",
        "health": "/health",
        "documents": "/documents",
        "retrieval": "/retrieval/search",
        "chat": "/chat",
        "summarize": "/summarize",
        "compare": "/compare",
        "research_gaps": "/research-gaps",
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
