"""Database row models."""

from datetime import datetime

from pydantic import BaseModel


class UserOut(BaseModel):
    id: int
    username: str
    credits: int
    is_admin: bool
    created_at: datetime | None = None
