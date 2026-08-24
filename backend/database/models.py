import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from backend.database.connection import Base

class UserModel(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    click_logs = relationship("ClickLogModel", back_populates="user")
    recommendations = relationship("RecommendationModel", back_populates="user")


class ProductModel(Base):
    __tablename__ = "products"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    description = Column(Text, nullable=True)
    category = Column(String, nullable=False, index=True)
    popularity = Column(Float, default=0.0)
    ctr = Column(Float, default=0.0)
    freshness = Column(Float, default=1.0)
    engagement = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    documents = relationship("DocumentModel", back_populates="product", cascade="all, delete-orphan")


class DocumentModel(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    raw_text = Column(Text, nullable=False)
    tokens_count = Column(Integer, default=0)
    features_vector = Column(JSON, nullable=True)  # Store precomputed feature fields
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    product = relationship("ProductModel", back_populates="documents")


class QueryModel(Base):
    __tablename__ = "queries"

    id = Column(String, primary_key=True, index=True)
    query_text = Column(String, unique=True, nullable=False, index=True)
    expanded_text = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class ClickLogModel(Base):
    __tablename__ = "click_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    product_id = Column(String, ForeignKey("products.id", ondelete="CASCADE"), nullable=False)
    query_text = Column(String, nullable=False)
    clicked_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("UserModel", back_populates="click_logs")


class RecommendationModel(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    algorithm_type = Column(String, nullable=False, index=True)  # hybrid, cf, cb, mf
    recommended_product_ids = Column(JSON, nullable=False)  # list of products
    scores = Column(JSON, nullable=False)  # matching confidence mapping
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("UserModel", back_populates="recommendations")


class ExperimentModel(Base):
    __tablename__ = "experiments"

    run_id = Column(String, primary_key=True)
    name = Column(String, index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    algorithm = Column(String, nullable=False)  # pointwise, pairwise_ranknet, listwise_lambdamart
    parameters = Column(JSON, nullable=False)
    metrics = Column(JSON, nullable=False)
    status = Column(String, nullable=False)  # SUCCESS, FAILED


class SearchQueryLogModel(Base):
    __tablename__ = "search_query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(String, nullable=False, index=True)
    algorithm_used = Column(String, nullable=False, default="listwise")
    weights_applied = Column(JSON, nullable=True)
    execution_time_ms = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

