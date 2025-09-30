import os

import httpx

from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    load_dotenv()
    llm: str = os.getenv("OPENAI_API_ENGINE")
    temperature: float = 0.
    api_key: str = os.getenv("OPENAI_API_KEY")
    endpoint: str = os.getenv("OPENAI_API_BASE")
    api_version: str = "2024-08-01-preview"
    time_out = httpx.Timeout(300.0, read=20.0, write=20.0, connect=5.0)
    max_tokens: Optional[int] = None
