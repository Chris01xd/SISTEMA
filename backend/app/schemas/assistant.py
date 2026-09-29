import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, StringConstraints


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    dataset_id: uuid.UUID
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
    # Previous turns of this conversation (the API is stateless); oldest first.
    history: list[ChatTurn] = Field(default_factory=list, max_length=10)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any]


class AskResponse(BaseModel):
    answer: str
    # Where the numbers in the answer came from: the read-only tools the model invoked.
    tool_calls: list[ToolCall]
    model: str


class AssistantStatus(BaseModel):
    available: bool
    model: str
    detail: str | None = None
