import matplotlib.pyplot as plt
import seaborn as sns
from utils.data_processing import compute_correlation_matrix
import numpy as np


def plot_correlation_matrix(df):
    try:
        correlation_matrix, _ = compute_correlation_matrix(df)
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title('Matrice di Correlazione tra PIL, Produzione, RLG e RM')
        plt.tight_layout()  # Aggiunto per migliorare la disposizione del grafico
        plt.show()
    except Exception as e:
        print(f"Errore durante la creazione della matrice di correlazione: {e}")



