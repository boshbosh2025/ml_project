import pytest
import os
from model_pipeline import prepare_data, train_model

def test_prepare_data():
    """Vérifie que prepare_data renvoie bien 4 éléments (X_train, X_test, y_train, y_test)."""
    # On utilise votre vrai fichier de données
    X_train, X_test, y_train, y_test = prepare_data("diabetic_data.csv")
    
    # Vérifications
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(y_train) == X_train.shape[0]
    assert len(y_test) == X_test.shape[0]

def test_train_model():
    """Vérifie que le modèle s'entraîne et retourne un Pipeline."""
    # Pour tester vite, on prend un petit échantillon de données
    X_train, X_test, y_train, y_test = prepare_data("diabetic_data.csv")
    
    # On entraîne sur le train set complet (ou une petite partie)
    model = train_model(X_train, y_train)
    
    # Vérifications
    assert model is not None
    assert hasattr(model, "predict")
