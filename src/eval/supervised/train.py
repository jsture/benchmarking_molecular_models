import numpy as np
import logging as log

from .models import get_clf_models, get_reg_models
from .const import CV_SPLITS, N_JOBS, VERBOSITY
from .utils import get_sklearn_scorer, multioutput_auroc_score
from ...common.types import EmbeddedDataset, HeadResult
from ..common.utils import get_train_data, get_test_data
from sklearn.model_selection import GridSearchCV
from typing import Tuple
from sklearn.metrics import roc_auc_score, make_scorer


def fit_model(X: np.ndarray, y: np.ndarray, 
              task: str, metric_name: str, 
              model_head: str, memory_weight: int,
              cv_verbosity: int = VERBOSITY,
              dataset_name: str = ""):
    tag = f"[{dataset_name}] [{model_head}] " if dataset_name else f"[{model_head}] "
    if task == "classification":
        # no_outputs = y.shape[1]
        no_outputs = y.shape[1] if len(y.shape) > 1 else 1
        models = get_clf_models(no_outputs, X.dtype)
    elif task == "regression":
        models = get_reg_models(X.dtype)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    
    if y.shape[1] > 1:
        log.info(f"{tag}Using multioutput AUROC scorer")
        scorer = make_scorer(multioutput_auroc_score, response_method='predict_proba')
    else:
        scorer = get_sklearn_scorer('roc_auc')

    y = np.nan_to_num(y, nan=0)

    log.info(f"{tag}Fitting on X={X.shape}, y={y.shape}")

    model = models[model_head]
    
    grid_search = GridSearchCV(
        model["model"],
        model["params"],
        cv=CV_SPLITS,
        scoring=scorer,
        n_jobs=int(N_JOBS / memory_weight),
        verbose=cv_verbosity,
        refit=True,
    )
    
    try:
        grid_search.fit(X, y)
    except ValueError as e:
        log.error(f"{tag}Error fitting model: {e}")
        if 'lbfgs' not in str(e):
            raise e
        log.error(f"{tag}L-BFG-S failed, replacing with SVD")
        if "clf__estimator_solver" in model["params"]:
            model["params"]["clf__estimator__solver"] = ["svd"]
        elif "clf__solver" in model["params"]:
            model["params"]["clf__solver"] = ["svd"]
        else:
            raise ValueError("Model parameters do not contain 'solver' or 'estimator__solver' key, cannot replace with SVD")
        grid_search = GridSearchCV(
            model["model"],
            model["params"],
            cv=CV_SPLITS,
            scoring=scorer,
            n_jobs=int(N_JOBS / memory_weight),
            verbose=cv_verbosity,
            refit=True,
        )
        grid_search.fit(X, y)  

    return {
        "model": model_head,
        "model_obj": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


def fit_and_eval_embedding(dataset: EmbeddedDataset, 
                           metric_name: str, model_head: str,
                           memory_weight: int,
                           cv_verbosity: int = VERBOSITY) -> HeadResult:
    X_train, y_train = get_train_data(dataset)
    best_model = fit_model(
        X=X_train, 
        y=y_train, 
        task=dataset.task, 
        metric_name=metric_name, 
        model_head=model_head,
        memory_weight=memory_weight,
        cv_verbosity=cv_verbosity,
        dataset_name=dataset.name,
    )
    X_test, y_test = get_test_data(dataset)
    log.info(f"[{dataset.name}] [{model_head}] Shapes: train={X_train.shape}, test={X_test.shape}")
    y_pred = best_model["model_obj"].predict_proba(X_test)

    return HeadResult(
        embedder=dataset.embedder,
        dataset_name=dataset.name,
        y_test_true=y_test,
        y_test_pred=y_pred,
        model=best_model["model"],
        hyperparams=best_model["best_params"],
        cv_score=best_model["best_score"],
    )
