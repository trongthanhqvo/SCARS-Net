from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pathlib import Path
import json


@dataclass
class XGBoostProbe:
    seed: int
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 20
    n_jobs: int = 4
    model: object | None = None
    classes_: np.ndarray | None = None

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        source_selection_features: np.ndarray,
        source_selection_labels: np.ndarray,
    ) -> "XGBoostProbe":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError(
                "The STFT+DCT baseline requires the pinned xgboost dependency"
            ) from error
        self.classes_ = np.unique(train_labels)
        if self.classes_.dtype.kind == "O":
            if not all(isinstance(value, str) for value in self.classes_):
                raise ValueError("Object labels must be strings for safe class serialization")
            self.classes_ = self.classes_.astype(str)
        mapping = {label: index for index, label in enumerate(self.classes_)}
        y_train = np.asarray([mapping[value] for value in train_labels], dtype=np.int64)
        y_selection = np.asarray(
            [mapping[value] for value in source_selection_labels], dtype=np.int64
        )
        if len(np.unique(y_selection)) != len(self.classes_):
            raise ValueError("source_selection must contain every recognition class")
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective="multi:softprob",
            num_class=len(self.classes_),
            eval_metric="mlogloss",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=self.n_jobs,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model.fit(
            np.asarray(train_features, dtype=np.float32),
            y_train,
            eval_set=[(np.asarray(source_selection_features, dtype=np.float32), y_selection)],
            verbose=False,
        )
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("XGBoostProbe is not fitted")
        return np.asarray(self.model.predict_proba(np.asarray(features, dtype=np.float32)))

    def save_model(self, path: Path) -> None:
        if self.model is None or self.classes_ is None:
            raise RuntimeError("XGBoostProbe is not fitted")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)
        np.save(path.with_suffix(path.suffix + ".classes.npy"), self.classes_, allow_pickle=False)

    def counted_cost(self) -> dict[str, int]:
        if self.model is None:
            raise RuntimeError("XGBoostProbe is not fitted")
        dumps = self.model.get_booster().get_dump(dump_format="json")

        def nodes(tree: dict) -> int:
            return 1 + sum(nodes(child) for child in tree.get("children", []))

        node_count = sum(nodes(json.loads(tree)) for tree in dumps)
        return {
            "parameters": int(node_count),
            "macs": int(len(dumps) * self.max_depth),
        }

    @classmethod
    def load_model(cls, path: Path, seed: int) -> "XGBoostProbe":
        try:
            from xgboost import XGBClassifier
        except ImportError as error:
            raise RuntimeError("Loading the XGBoost baseline requires xgboost") from error
        instance = cls(seed=seed)
        instance.model = XGBClassifier()
        instance.model.load_model(path)
        instance.classes_ = np.load(path.with_suffix(path.suffix + ".classes.npy"), allow_pickle=False)
        return instance
