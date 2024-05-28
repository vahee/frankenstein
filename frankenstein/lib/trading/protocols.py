from typing import Protocol, Optional
from datetime import datetime
import pandas as pd


class IDataProvider(Protocol):
    def ticks(self, symbol: str, timestamp: Optional[datetime], max_ticks: Optional[int] = None) -> pd.DataFrame:
        ...
    
    def bars(self, symbol: str, timeframe: str, timestamp: Optional[datetime] = None, max_bars: Optional[int] = None) -> pd.DataFrame:
        ...
        
    def ask(self, symbol: str, timestamp: Optional[datetime] = None) -> float:
        ...
        
    def bid(self, symbol: str, timestamp: Optional[datetime] = None) -> float:
        ...
        
    def get_time(self) -> datetime:
        ...
        
    def step(self) -> None:
        ...
        
    def reset(self, start: str, end: str, freq: str) -> None:
        ...