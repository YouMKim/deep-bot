from dataclasses import dataclass
from dotenv import load_dotenv 
import os
from typing import Optional, Dict, Any
from enum import Enum


load_dotenv() 

class AIProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


@dataclass
class AIConfig:
    model_name: str  
    model_temperature: float = 0.7
    model_max_tokens: int = 500
    max_retries: int = 3
    timeout_seconds: int = 30
    rate_limit_per_minute: int = 100 

    open_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthopic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def from_openai(cls, usage) -> "TokenUsage":
        return cls(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens
        )
    
    @classmethod
    def from_anthropic(cls, input_tokens: int, output_tokens: int) -> "TokenUsage":
        return cls(
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens
        )


@dataclass
class CostDetails:
    provider: str
    model: str
    input_cost: float
    output_cost: float
    total_cost: float


@dataclass
class AIRequest:
    prompt: str  # The prompt text to send
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    provider: Optional[str] = None 
    metadata: Optional[Dict[str, Any]] = None

    def validate_request(self):
        if not self.prompt or not self.prompt.strip():
            raise ValueError("Prompt cannot be empty")
        if self.max_tokens and self.max_tokens <= 0:
            raise ValueError("Max tokens must be greater than 0")
        if self.temperature and (self.temperature < 0 or self.temperature > 2):
            raise ValueError("Temperature must be between 0 and 2")


@dataclass
class AIResponse:
    content: str  
    model: str  
    usage: Optional[TokenUsage] = None
    cost: Optional[CostDetails] = None
    latency_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
