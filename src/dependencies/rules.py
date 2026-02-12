from dataclasses import dataclass
from typing import List, Callable


@dataclass
class DependencyRule:
    target: str
    sources: List[str]
    func: Callable