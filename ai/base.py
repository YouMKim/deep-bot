from abc import ABC, abstractmethod
from typing import Optional
from .models import AIRequest, AIResponse, TokenUsage, CostDetails


class BaseAIProvider(ABC):
    
    @abstractmethod
    async def complete(self, request: AIRequest) -> AIResponse:
        pass
    
    @abstractmethod
    def calculate_cost(self, model: str, usage: TokenUsage) -> CostDetails:
        pass
    
    @abstractmethod
    def supports_model(self, model: str) -> bool:
        pass
    
    @abstractmethod
    def get_default_model(self) -> str:
        pass
    
    def validate_request(self, request: AIRequest) -> None:
        request.validate_request()
        
        if not self.supports_model(request.model or self.get_default_model()):
            raise ValueError(
                f"Model {request.model} is not supported by this provider"
            )

