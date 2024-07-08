# File: utils/data_processing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_correlation_matrix(df):
    # Filtraggio dei dati
    dati_filtrati_pil = df[df['TIPO_DATO_PIL_SEC2010'] == 'B1GQ_B_W2_S1_X2']
    dati_filtrati_produzione = df[df['TIPO_DATO_PIL_SEC2010'] == 'P1_C_W2_S1']
    dati_filtrati_rlg = df[df['TIPO_DATO_PIL_SEC2010'] == 'P3_D_W0_S1']
    dati_filtrati_rm = df[df['TIPO_DATO_PIL_SEC2010'] == 'B2A3G_B_W2_S1']

    # Aggregazione per somma o media, a seconda del caso
    dati_aggregati_pil = dati_filtrati_pil.groupby('TIME')['Value'].sum().reset_index(name='Value_pil')
    dati_aggregati_produzione = dati_filtrati_produzione.groupby('TIME')['Value'].mean().reset_index(name='Value_produzione')
    dati_aggregati_rlg = dati_filtrati_rlg.groupby('TIME')['Value'].sum().reset_index(name='Value_rlg')
    dati_aggregati_rm = dati_filtrati_rm.groupby('TIME')['Value'].sum().reset_index(name='Value_rm')

    # Unione dei DataFrame in uno singolo
    merged_df = pd.merge(dati_aggregati_pil, dati_aggregati_produzione, on='TIME', how='outer')
    merged_df = pd.merge(merged_df, dati_aggregati_rlg, on='TIME', how='outer')
    merged_df = pd.merge(merged_df, dati_aggregati_rm, on='TIME', how='outer')

    # Rimozione delle righe con valori mancanti (NaN)
    merged_df.dropna(inplace=True)

    # Calcolo della matrice di correlazione
    numeric_columns = ['Value_pil', 'Value_produzione', 'Value_rlg', 'Value_rm']
    correlation_matrix = merged_df[numeric_columns].corr()

    return correlation_matrix, merged_df

def plot_correlation_heatmap(correlation_matrix):
    # Visualizzazione della matrice di correlazione tramite heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione tra PIL, Produzione, RLG e RM')
    plt.show()
