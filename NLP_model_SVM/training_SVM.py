from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
import numpy as np


def build_vectorizer() -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),  # 只使用一个字～三个字的词汇
        min_df=5,  # <5 -> outlier
    )
    return vectorizer


def tune_linear_svm_with_cv(
    texts: List[str],
    labels: List[str],
    search_mode: str = "random",
    cv: int = 5,
    n_iter: int = 20,
) -> Dict[str, Any]:
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)

    # model
    base_clf = LinearSVC()

    # random search
    param_dist = {
        "C": np.logspace(-3, 2, 100)  # 0.001 ~ 100, randomized 100 values
    }

    searcher = RandomizedSearchCV(
        estimator=base_clf,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,  # CPU usage
        random_state=42,
    )

    searcher.fit(X, labels)

    best_clf = searcher.best_estimator_
    best_params = searcher.best_params_
    best_score = searcher.best_score_

    return {
        "best_model": best_clf,
        "vectorizer": vectorizer,
        "best_params": best_params,
        "best_score": best_score,
        "cv_results": searcher.cv_results_,
    }
