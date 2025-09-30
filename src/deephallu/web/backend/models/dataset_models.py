from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class DatasetInfo(BaseModel):
    name: str
    type: str
    description: str
    path: str
    categories: Optional[List[str]] = None
    total_samples: Optional[int] = None
    last_updated: Optional[datetime] = None


class DatasetSample(BaseModel):
    id: str
    image_name: str
    image_path: str
    category: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DatasetListResponse(BaseModel):
    datasets: List[DatasetInfo]
    total_count: int


class DatasetSamplesResponse(BaseModel):
    samples: List[DatasetSample]
    total_count: int
    page: int
    page_size: int
    has_next: bool
    has_prev: bool


class DatasetBrowseRequest(BaseModel):
    dataset_name: str
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    category: Optional[str] = None
    search_query: Optional[str] = None


class DatasetStatsResponse(BaseModel):
    dataset_name: str
    total_samples: int
    categories: Optional[Dict[str, int]] = None
    sample_distribution: Optional[Dict[str, Any]] = None