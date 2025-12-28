# test_titanic_ml.py
# by: NathanGr33n
# Unit tests for titanic_ml.py functions using pytest and monkeypatching.
# These tests mock out cross-validation and model training to verify
# correct behavior without relying on actual model implementations.

import pandas as pd
import numpy as np
import pytest

from titanic_ml import evaluate_models_cv, train_best_model


@pytest.fixture
def sample_data():
    #Create a small, simple binary classification dataset.
    X = pd.DataFrame(
        {
            "feature1": [0.0, 0.0, 1.0, 1.0],
            "feature2": [0.0, 1.0, 0.0, 1.0],
        }
    )
    y = pd.Series([0, 0, 1, 1])
    return X, y


def test_evaluate_models_cv_returns_mean_accuracy_for_both_models(sample_data, monkeypatch):
    #evaluate_models_cv should perform CV and return mean accuracy for LR and RF.

    X, y = sample_data

    # Use deterministic fake scores so we can check the mean logic
    scores_lr = np.array([0.8, 0.9, 1.0])
    scores_rf = np.array([0.7, 0.6, 0.8])

    def fake_cross_val_score(model, X_arg, y_arg, cv=None, scoring=None, n_jobs=None):
        # Decide which fake scores to return based on the model type
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline

        if isinstance(model, Pipeline) and isinstance(model.named_steps["clf"], LogisticRegression):
            return scores_lr
        if isinstance(model, RandomForestClassifier):
            return scores_rf
        raise AssertionError("Unexpected model passed to cross_val_score in test.")

    monkeypatch.setattr("titanic_ml.cross_val_score", fake_cross_val_score)

    results = evaluate_models_cv(X, y, cv_splits=3)

    assert set(results.keys()) == {"Logistic Regression", "Random Forest"}
    assert results["Logistic Regression"] == pytest.approx(scores_lr.mean())
    assert results["Random Forest"] == pytest.approx(scores_rf.mean())


def test_evaluate_models_cv_uses_provided_cv_splits(sample_data, monkeypatch):
    #evaluate_models_cv should respect the cv_splits argument when constructing StratifiedKFold.

    X, y = sample_data

    created_cv_objects = []

    class DummyCV:
        pass

    def fake_stratified_kfold(*, n_splits, shuffle, random_state):
        created = DummyCV()
        created.n_splits = n_splits
        created.shuffle = shuffle
        created.random_state = random_state
        created_cv_objects.append(created)
        return created

    def fake_cross_val_score(model, X_arg, y_arg, cv=None, scoring=None, n_jobs=None):
        # Return simple constant scores; we only care about n_splits usage here
        return np.array([0.5, 0.5, 0.5])

    monkeypatch.setattr("titanic_ml.StratifiedKFold", fake_stratified_kfold)
    monkeypatch.setattr("titanic_ml.cross_val_score", fake_cross_val_score)

    # First with 3 splits
    evaluate_models_cv(X, y, cv_splits=3)
    assert created_cv_objects[-1].n_splits == 3

    # Then with 7 splits
    evaluate_models_cv(X, y, cv_splits=7)
    assert created_cv_objects[-1].n_splits == 7
    # Also ensure shuffle/random_state are wired through correctly
    assert created_cv_objects[-1].shuffle is True
    assert created_cv_objects[-1].random_state == 42


def test_train_best_model_logistic_regression_with_scaling(sample_data):
    #train_best_model should return a fitted LogisticRegression and StandardScaler.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X, y = sample_data

    model, scaler = train_best_model(X, y, "Logistic Regression")

    assert isinstance(model, LogisticRegression)
    assert isinstance(scaler, StandardScaler)

    # Scaler should be fitted to the right number of features
    assert hasattr(scaler, "mean_")
    assert scaler.mean_.shape[0] == X.shape[1]

    # Model should be fitted (has coefficients) and able to predict
    assert hasattr(model, "coef_")
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled)
    assert set(preds).issubset({0, 1})


def test_train_best_model_random_forest_feature_importance(sample_data):
    #train_best_model should train a RandomForest and expose feature_importances_.
    from sklearn.ensemble import RandomForestClassifier

    X, y = sample_data

    model, scaler = train_best_model(X, y, "Random Forest")

    assert isinstance(model, RandomForestClassifier)
    # For RF branch, scaler should be None
    assert scaler is None

    # Model should expose feature importances for all features
    assert hasattr(model, "feature_importances_")
    assert len(model.feature_importances_) == X.shape[1]


def test_train_best_model_raises_for_unknown_model(sample_data):
    #An unknown model name should raise a ValueError.
    X, y = sample_data

    with pytest.raises(ValueError, match="Unknown model name: SVM"):
        train_best_model(X, y, "SVM")
