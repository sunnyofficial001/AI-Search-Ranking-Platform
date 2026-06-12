from typing import List, Optional

from sqlalchemy.orm import Session

from backend.database.models import (
    ClickLogModel,
    DocumentModel,
    ExperimentModel,
    ProductModel,
    QueryModel,
    RecommendationModel,
    UserModel,
)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, user_id: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.id == user_id).first()

    def get_all(self) -> List[UserModel]:
        return self.db.query(UserModel).all()

    def create(self, user_id: str, username: str, email: str) -> UserModel:
        user = UserModel(id=user_id, username=username, email=email)
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user


class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, prod_id: str) -> Optional[ProductModel]:
        return self.db.query(ProductModel).filter(ProductModel.id == prod_id).first()

    def get_all(self) -> List[ProductModel]:
        return self.db.query(ProductModel).all()

    def create(
        self,
        id: str,
        title: str,
        description: str,
        category: str,
        popularity: float,
        ctr: float,
        freshness: float,
        engagement: float,
    ) -> ProductModel:
        p = ProductModel(
            id=id,
            title=title,
            description=description,
            category=category,
            popularity=popularity,
            ctr=ctr,
            freshness=freshness,
            engagement=engagement,
        )
        self.db.add(p)
        self.db.commit()
        self.db.refresh(p)
        return p


class DocumentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, doc_id: str) -> Optional[DocumentModel]:
        return self.db.query(DocumentModel).filter(DocumentModel.id == doc_id).first()

    def get_by_product_id(self, prod_id: str) -> Optional[DocumentModel]:
        return self.db.query(DocumentModel).filter(DocumentModel.product_id == prod_id).first()

    def create(
        self,
        doc_id: str,
        product_id: str,
        raw_text: str,
        tokens_count: int,
        features_vector: dict,
    ) -> DocumentModel:
        doc = DocumentModel(
            id=doc_id,
            product_id=product_id,
            raw_text=raw_text,
            tokens_count=tokens_count,
            features_vector=features_vector,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc


class QueryRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_text(self, text: str) -> Optional[QueryModel]:
        return self.db.query(QueryModel).filter(QueryModel.query_text == text).first()

    def save(self, id: str, text: str, expanded_text: str) -> QueryModel:
        q = self.get_by_text(text)
        if q:
            q.expanded_text = expanded_text
            self.db.commit()
            self.db.refresh(q)
            return q
        q = QueryModel(id=id, query_text=text, expanded_text=expanded_text)
        self.db.add(q)
        self.db.commit()
        self.db.refresh(q)
        return q


class ClickLogRepository:
    def __init__(self, db: Session):
        self.db = db

    def log_click(self, user_id: str, product_id: str, query_text: str) -> ClickLogModel:
        log = ClickLogModel(user_id=user_id, product_id=product_id, query_text=query_text)
        self.db.add(log)
        self.db.commit()
        self.db.refresh(log)
        return log


class RecommendationRepository:
    def __init__(self, db: Session):
        self.db = db

    def save_recs(self, user_id: str, rec_type: str, prod_ids: List[str], scores: List[float]) -> RecommendationModel:
        rec = RecommendationModel(
            user_id=user_id,
            algorithm_type=rec_type,
            recommended_product_ids=prod_ids,
            scores=scores,
        )
        self.db.add(rec)
        self.db.commit()
        self.db.refresh(rec)
        return rec


class ExperimentRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_run_id(self, run_id: str) -> Optional[ExperimentModel]:
        return self.db.query(ExperimentModel).filter(ExperimentModel.run_id == run_id).first()

    def get_all(self) -> List[ExperimentModel]:
        return self.db.query(ExperimentModel).order_by(ExperimentModel.timestamp.desc()).all()

    def create(
        self,
        run_id: str,
        name: str,
        algorithm: str,
        parameters: dict,
        metrics: dict,
        status: str,
    ) -> ExperimentModel:
        exp = ExperimentModel(
            run_id=run_id,
            name=name,
            algorithm=algorithm,
            parameters=parameters,
            metrics=metrics,
            status=status,
        )
        self.db.add(exp)
        self.db.commit()
        self.db.refresh(exp)
        return exp

    def delete(self, run_id: str) -> bool:
        exp = self.get_by_run_id(run_id)
        if exp:
            self.db.delete(exp)
            self.db.commit()
            return True
        return False
