from pydantic import BaseModel, EmailStr, ConfigDict
from typing import List, Dict, Any, Optional
import datetime

# User Schemas
class UserBase(BaseModel):
    id: str
    username: str
    email: EmailStr

class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)
    created_at: datetime.datetime

# Product Schemas
class ProductBase(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    category: str
    popularity: float
    ctr: float
    freshness: float
    engagement: float

class ProductResponse(ProductBase):
    model_config = ConfigDict(from_attributes=True)
    created_at: datetime.datetime

# Search Schemas
class SearchRequest(BaseModel):
    query: str
    weights: Optional[Dict[str, float]] = None

class SearchResultItem(BaseModel):
    productId: str
    title: str
    category: str
    scores: Dict[str, float]  # pointwise, pairwise, listwise scores
    originalRank: int
    finalRank: int
    relevanceLabel: int
    features: Dict[str, float]

class SearchResponse(BaseModel):
    query: str
    expandedQuery: str
    results: List[SearchResultItem]

# Recommendation Schemas
class RecommendRequest(BaseModel):
    userId: Optional[str] = "user-1"
    productId: Optional[str] = None
    type: str  # content, collaborative, matrix_factorization, hybrid
    hybridWeight: Optional[float] = 0.5
    epochs: Optional[int] = 30
    latentDim: Optional[int] = 3
    lr: Optional[float] = 0.05

class RecommendItem(BaseModel):
    productId: str
    title: str
    category: str
    score: float
    type: str
    breakdown: str

class RecommendResponse(BaseModel):
    type: str
    results: List[RecommendItem]
    losses: Optional[List[Dict[str, float]]] = None

# Explainability Schemas
class ExplainRequest(BaseModel):
    productId: str
    query: str

class ContributionItem(BaseModel):
    feature: str
    value: float
    shapleyValue: float

class ExplainResponse(BaseModel):
    productId: str
    productTitle: str
    baseValue: float
    finalScore: float
    contributions: List[ContributionItem]

# Experiment Tracking Schemas
class ExperimentRunRequest(BaseModel):
    algorithm: str
    learningRate: Optional[float] = None
    epochs: Optional[int] = None
    nEstimators: Optional[int] = None

class ExperimentRunResponse(BaseModel):
    runId: str
    name: str
    timestamp: str
    algorithm: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: str

class ExperimentDashboardResponse(BaseModel):
    runs: List[ExperimentRunResponse]
    modelRegistry: List[Dict[str, Any]]
