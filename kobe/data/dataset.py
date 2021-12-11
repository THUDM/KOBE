from dataclasses import dataclass
from typing import List


@dataclass
class Example:
    title_token_ids: List[int]
    description_token_ids: List[str]
    condition_token_ids: List[str]
    fact_token_ids: List[str]
    description: str
