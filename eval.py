"""
Script de evaluación del modelo
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_model_and_data():
    """
    Carga el modelo entrenado y los datos de prueba
    """
    print("📂 Cargando modelo y datos...")

    model = joblib.load("Model/drug_classifier.pkl")
    label_encoder = joblib.load("Model/label_encoder.pkl")

    X_test = pd.read_csv("data/X_test.csv")
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    return model, label_encoder, X_test, y_test


def plot_confusion_matrix(y_true, y_pred, label_encoder):
    """
    Genera y guarda la matriz de confusión
    """
    print("📊 Generando matriz de confusión...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
    )
    plt.title("Confusion Matrix - Drug Classification", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()

    # Guardar imagen
    os.makedirs("Results", exist_ok=True)
    plt.savefig("Results/model_results.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("✅ Matriz de confusión guardada en Results/model_results.png")


def save_metrics(y_true, y_pred, label_encoder):
    """
    Calcula y guarda las métricas del modelo
    """
    print("📈 Calculando métricas...")

    # Calcular métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Reporte de clasificación
    report = classification_report(y_true, y_pred, target_names=label_encoder.classes_)

    # Guardar métricas en archivo de texto
    os.makedirs("Results", exist_ok=True)
    with open("Results/metrics.txt", "w") as f:
        f.write("# Drug Classification Model - Evaluation Metrics\n\n")
        f.write(f"**Accuracy**: {accuracy:.4f}\n")
        f.write(f"**Precision**: {precision:.4f}\n")
        f.write(f"**Recall**: {recall:.4f}\n")
        f.write(f"**F1-Score**: {f1:.4f}\n\n")
        f.write("## Classification Report\n\n")
        f.write("```\n")
        f.write(report)
        f.write("\n```\n")

    print("✅ Métricas guardadas en Results/metrics.txt")

    # Imprimir métricas
    print("\n" + "=" * 50)
    print("📊 RESULTADOS DE EVALUACIÓN")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 50 + "\n")

    return accuracy, precision, recall, f1


def evaluate_model():
    """
    Evalúa el modelo entrenado
    """
    print("🎯 Iniciando evaluación del modelo...")

    # Cargar modelo y datos
    model, label_encoder, X_test, y_test = load_model_and_data()

    # Hacer predicciones
    print("🔮 Realizando predicciones...")
    y_pred = model.predict(X_test)

    # Generar matriz de confusión
    plot_confusion_matrix(y_test, y_pred, label_encoder)

    # Calcular y guardar métricas
    save_metrics(y_test, y_pred, label_encoder)

    print("✨ Evaluación completada exitosamente!")


if __name__ == "__main__":
    evaluate_model()
