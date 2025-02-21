import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# ===========================================
# Model Training
# ===========================================

def train_classification_model(model, X_train, y_train):
    print("Training the model...")
    model.fit(X_train, y_train)
    print("Model training completed.")
    return model

# ===========================================
# Model Evaluation
# ===========================================

def evaluate_classification_model(model, X_test, y_test, y_proba=None, threshold=0.5):
    """
    Evaluate the classification model's performance.
    """
    if y_proba is None:
        y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_proba))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
def fit_and_evaluate_model(model_name, model):
    """
    Fit the GARCH model and return the result.
    """
    print(f"\nFitting {model_name}...")
    try:
        result = model.fit(
            update_freq=5,
            disp='off',
            tol=1e-6,
            options={'maxiter': 1000}
        )
        print(f"\n{model_name} Summary:")
        print(result.summary())
        return result
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        return None
    
# ===========================================
# Model Persistence (Save/Load)
# ===========================================

def save_model(model, filepath):
    """
    Save the trained model to a file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a model from a file.
    """
    return joblib.load(filepath)

def save_garch_model(model_result, filepath):
    """
    Save the GARCH model result to a file using pickle.
    """
    import pickle
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_result, f)

def load_garch_model(filepath):
    """
    Load a GARCH model result from a file using pickle.
    """
    import pickle
    with open(filepath, 'rb') as f:
        model_result = pickle.load(f)
    return model_result

# ===========================================
# Hyperparameter Tuning
# ===========================================

def perform_hyperparameter_tuning(estimator, param_grid, X_train, y_train, scoring='roc_auc', random_state=42, verbose=1):
    """
    Perform hyperparameter tuning using GridSearchCV for any estimator.
    """
    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        n_jobs=-1,
        verbose=verbose
    )
    
    # Fit the model before accessing best parameters and score
    grid_search.fit(X_train, y_train)

    # Access attributes after the fit
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best score: {grid_search.best_score_}")

    return grid_search.best_estimator_

# ===========================================
# Feature Importance and Visualization
# ===========================================

def plot_feature_importance(model, feature_names, model_type='tree'):
    """
    Plot the feature importance or coefficients from the model.
    """
    feature_importance_df = pd.DataFrame()

    if model_type == 'tree':
        # Tree-based models like Random Forest
        feature_importance_df['Feature'] = feature_names
        feature_importance_df['Importance'] = model.feature_importances_
    elif model_type == 'logistic_regression':
        # Logistic Regression coefficients
        feature_importance_df['Feature'] = feature_names
        feature_importance_df['Importance'] = model.coef_[0]
    else:
        print(f"Unsupported model type: {model_type}")
        return

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.title(f'Feature Importance ({model_type})')
    plt.tight_layout()
    plt.show()