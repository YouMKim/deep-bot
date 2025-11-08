from enum import Enum
from typing import List


class ChunkStrategy(str, Enum):
    SINGLE = "single"
    TEMPORAL = "temporal"
    CONVERSATION = "conversation"
    SLIDING_WINDOW = "sliding_window"
    AUTHOR = "author"
    TOKENS = "tokens"
    TEMPORAL_WITH_TOKEN_LIMIT = "temporal_with_token_limit"

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]

