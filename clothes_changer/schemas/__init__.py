"""Pydantic request/response schemas."""

from datetime import datetime

from pydantic import BaseModel, Field


class UserCredentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1)


class UserOut(BaseModel):
    id: int
    username: str
    credits: int
    is_admin: bool
    created_at: datetime | None = None


class AuthResponse(BaseModel):
    success: bool
    message: str
    user: UserOut | None = None


class SegmentResponse(BaseModel):
    overlay_base64: str
    person_mask_base64: str
    clothes_mask_base64: str


class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = ""
    steps: int = Field(default=30, ge=10, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    seed: int | None = None
    model: str | None = None
    use_controlnet: bool | None = None


class GenerateResponse(BaseModel):
    output_base64: str
    filename: str
    credits_remaining: int


class HistoryItem(BaseModel):
    id: int
    filename: str | None
    prompt: str | None
    created_at: str


class ModelInfo(BaseModel):
    id: str
    name: str
    source: str  # "huggingface" | "local"
