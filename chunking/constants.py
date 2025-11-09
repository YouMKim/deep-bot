from enum import Enum
from typing import List


class ChunkStrategy(str, Enum):
    SINGLE = "single"
    TEMPORAL = "temporal"
    CONVERSATION = "conversation"
    SLIDING_WINDOW = "sliding_window"
    AUTHOR = "author"
    TOKENS = "tokens"

    @classmethod
    def values(cls) -> List[str]:
        return [member.value for member in cls]

