"""
Script de entrenamiento del modelo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os


def create_sample_data():
    """
    Crea datos de ejemplo para Drug Classification
    """
    np.random.seed(42)
    n_samples = 1000

    # Crear características
    data = {
        "Age": np.random.randint(20, 80, n_samples),
        "Sex": np.random.choice(["M", "F"], n_samples),
        "BP": np.random.choice(["HIGH", "NORMAL", "LOW"], n_samples),
        "Cholesterol": np.random.choice(["HIGH", "NORMAL"], n_samples),
        "Na_to_K": np.random.uniform(5, 40, n_samples),
        "Drug": np.random.choice(
            ["DrugA", "DrugB", "DrugC", "DrugX", "DrugY"], n_samples
        ),
    }

    df = pd.DataFrame(data)
    return df


def preprocess_data(df):
    """
    Preprocesa los datos
    """
    # Codificar variables categóricas
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["BP"] = le_bp.fit_transform(df["BP"])
    df["Cholesterol"] = le_chol.fit_transform(df["Cholesterol"])
    df["Drug"] = le_drug.fit_transform(df["Drug"])

    return df, le_drug


def train_model():
    """
    Entrena el modelo de clasificación de medicamentos
    """
    print(" Iniciando entrenamiento del modelo...")

    # Crear carpetas necesarias
    os.makedirs("data", exist_ok=True)
    os.makedirs("Model", exist_ok=True)

    # Crear o cargar datos
    print(" Cargando datos...")
    df = create_sample_data()

    # Guardar datos originales
    df.to_csv("data/drug_data.csv", index=False)
    print(f" Datos guardados: {len(df)} muestras")

    # Preprocesar datos
    print(" Preprocesando datos...")
    df_processed, label_encoder = preprocess_data(df)

    # Separar características y target
    X = df_processed.drop("Drug", axis=1)
    y = df_processed["Drug"]

    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Guardar datos de prueba para evaluación
    X_test.to_csv("data/X_test.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    # Entrenar modelo
    print(" Entrenando modelo Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Guardar modelo y label encoder
    print(" Guardando modelo...")
    joblib.dump(model, "Model/drug_classifier.pkl")
    joblib.dump(label_encoder, "Model/label_encoder.pkl")

    # Calcular accuracy en training
    train_accuracy = model.score(X_train, y_train)
    print(f" Accuracy en entrenamiento: {train_accuracy:.4f}")

    print("✨ Entrenamiento completado exitosamente!")

    return model


if __name__ == "__main__":
    train_model()
