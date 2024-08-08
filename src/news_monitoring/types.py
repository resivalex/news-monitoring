from typing import TypedDict
import numpy as np


# TypedDict for representing a news message structure
class NewsMessage(TypedDict):
    published_at: str
    content: str
    url: str


class MessageEmbedding(TypedDict):
    embedding: np.ndarray
