from enum import Enum, StrEnum
from typing import Any, Callable, Dict, List, Optional
from SharedKernel.utils.yamlenv import load_env_yaml
from pydantic import BaseModel, Field

config = load_env_yaml()

class PromptType(StrEnum):
    NONE = "none"
    STREAM = "stream"

class ChatRequest(BaseModel):
    message: str

class ChatMessageRequest(BaseModel):
    session_id: str = "user1234"
    message: str

class TemplateType(str, Enum):
    from_template = "from_template"
    prompt_template = "prompt_template"
    chat_template = "chat_template"
    message_placeholder = "message_placeholder"

class ChatTemplateRequest(BaseModel):
    message: str
    template: TemplateType

class TechType(str, Enum):
    ZERO = "zero_shot"
    FEW = "few_shot"
    COT = "CoT"
    REACT = "ReAct"

class ChatTechniqueRequest(BaseModel):
    message: str
    tech: TechType

class StructedOutputType(str, Enum):
    none = "none"
    stream = "stream"

class MemoryType(str, Enum):
    short = "short",
    long = "long"

class YouTubeVideo(BaseModel):
    title: str = Field(description="Tiêu đề video")
    channel: str = Field(description="Tên kênh YouTube")
    views: int = Field(description="Số lượt xem")
    upload_date: str = Field(description="Ngày đăng video")
    is_short: bool = Field(description="Video có phải YouTube Shorts hay không")

class Callback(BaseModel):
    ainvoke: Callable[[Any], Any]
    astream: Callable[[Any], Any]

class SplitRequest:
    text: str
    chunk_size: int
    chunk_overlap: int 
    separators: Optional[List[str]]

class ChunkResponse(BaseModel):
    index: int
    content: str
    length: int
    metadata: Optional[Dict[str, Any]] = None
