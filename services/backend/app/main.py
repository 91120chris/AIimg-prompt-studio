from fastapi import FastAPI

from app.api.agent import router as agent_router
from app.api.files import router as files_router
from app.api.generation import router as generation_router
from app.api.providers import router as providers_router
from app.api.security import router as security_router
from app.api.sessions import router as sessions_router
from app.api.settings import router as settings_router
from app.core.app_settings_store import load_persisted_app_settings
from app.cors import configure_cors
from app.db.session import create_db_engine, init_db
from app.schemas import HealthResponse
from app.settings import Settings, get_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    app = FastAPI(title=app_settings.app_name, version=app_settings.app_version)
    app.state.settings = app_settings
    app.state.engine = create_db_engine(app_settings)
    init_db(app.state.engine)
    if settings is None or app_settings.load_persisted_settings:
        load_persisted_app_settings(app.state.engine, app_settings)
    configure_cors(app, app_settings)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            app_name=app_settings.app_name,
            version=app_settings.app_version,
            environment=app_settings.app_env,
        )

    app.include_router(settings_router)
    app.include_router(security_router)
    app.include_router(providers_router)
    app.include_router(sessions_router)
    app.include_router(agent_router)
    app.include_router(generation_router)
    app.include_router(files_router)

    return app


app = create_app()
