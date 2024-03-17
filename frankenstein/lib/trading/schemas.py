from dataclasses import dataclass
from typing import Optional
from datetime import datetime

@dataclass
class Signal:
    timestamp: datetime
    direction: int
    tp_pips: Optional[int]
    sl_pips: Optional[int]
    comment: str
    symbol: str