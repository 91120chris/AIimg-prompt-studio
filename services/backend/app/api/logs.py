from fastapi import APIRouter, Query, Request
from sqlmodel import select

from app.db.models import LogRecord
from app.db.session import new_session
from app.schemas.logs import LogResponse

router = APIRouter(prefix="/logs", tags=["logs"])


def _engine(request: Request):
    return request.app.state.engine


@router.get("", response_model=list[LogResponse])
def list_logs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[LogResponse]:
    with new_session(_engine(request)) as db:
        records = db.exec(select(LogRecord).order_by(LogRecord.created_at.desc()).limit(limit)).all()
        return [
            LogResponse(
                log_id=record.log_id,
                level=record.level,
                message=record.message,
                created_at=record.created_at,
            )
            for record in records
        ]
