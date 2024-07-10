# File: utils/data_processing.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def compute_correlation_matrix(df1):
    # Filtraggio dei dati
    
    Tipo_di_dato_pil = ["B1GQ_B_W2_S1_X2"]
    Ed_pil = ["Set-2023"]
    prezzi = ['prezzi correnti']
    dati_filtrati_pil = df1[(df1['TIPO_DATO_PIL_SEC2010'].isin(Tipo_di_dato_pil)) &
                        (df1['Edizione'].isin(Ed_pil))&
                        (df1['Valutazione'].isin(prezzi))]

# Eliminazione delle colonne non necessarie per il PIL
    colonne_da_rimuovere_pil = ['Seleziona periodo', 'Flag Codes', 'Flags']
    dati_completi_pil = dati_filtrati_pil.drop(columns=colonne_da_rimuovere_pil, axis=1)

# Lettura del dataset Produzione
    Tipo_di_dato_prod = ["P1_C_W2_S1"]
    Ed_prod = ["Set-2023"]
    prezzi = ['prezzi correnti']
    dati_filtrati_prod = df1[(df1['TIPO_DATO_PIL_SEC2010'].isin(Tipo_di_dato_prod)) &
                             (df1['Edizione'].isin(Ed_prod))&
                             (df1['Valutazione'].isin(prezzi))]

# Eliminazione delle colonne non necessarie per la Produzione
    colonne_da_rimuovere_prod = ['Seleziona periodo', 'Flag Codes', 'Flags']
    dati_completi_prod = dati_filtrati_prod.drop(columns=colonne_da_rimuovere_prod, axis=1)

# Dati da cercare all'interno del DataFrame.
    Tipo_di_dato = ["P3_D_W0_S1"]
    Ed = ["Set-2023"]
    prezzi = ['prezzi correnti']

# Filtra il DataFrame df1 in base ai criteri specificati.
    dati_filtrati_sp = df1[(df1['TIPO_DATO_PIL_SEC2010'].isin(Tipo_di_dato)) &
                            (df1['Edizione'].isin(Ed))&
                            (df1['Valutazione'].isin(prezzi))]

# Concatena le righe filtrate in un unico DataFrame.
    dati_completi_3 = pd.concat([dati_filtrati_sp])

# Eliminiamo le colonne del datbase che non servono
    colonne_da_rimuovere = ['Seleziona periodo', 'Flag Codes', 'Flags']
    dati_completi_sp = dati_filtrati_sp.drop(columns=colonne_da_rimuovere, axis=1)

# Dati da cercare all'interno del DataFrame.
    Tipo_di_dato = ["B2A3G_B_W2_S1"]
    Ed = ["Set-2023"]
    prezzi = ['prezzi correnti']

# Filtra il DataFrame df1 in base ai criteri specificati.
    dati_filtrati_rm= df1[(df1['TIPO_DATO_PIL_SEC2010'].isin(Tipo_di_dato)) &
                         (df1['Edizione'].isin(Ed))&
                         (df1['Valutazione'].isin(prezzi))]

# Concatena le righe filtrate in un unico DataFrame.
    dati_completi_4 = pd.concat([dati_filtrati_rm])

# Eliminiamo le colonne del datbase che non servono
    colonne_da_rimuovere = ['Seleziona periodo', 'Flag Codes', 'Flags']
    dati_completi_rm = dati_filtrati_rm.drop(columns=colonne_da_rimuovere, axis=1)

    # Aggregazione per somma o media, a seconda del caso
    dati_aggregati_pil = dati_filtrati_pil.groupby('TIME')['Value'].sum().reset_index(name='Value_pil')
    dati_aggregati_produzione = dati_filtrati_prod.groupby('TIME')['Value'].mean().reset_index(name='Value_produzione')
    dati_aggregati_sp = dati_filtrati_sp.groupby('TIME')['Value'].sum().reset_index(name='Value_sp')
    dati_aggregati_rm = dati_filtrati_rm.groupby('TIME')['Value'].sum().reset_index(name='Value_rm')

    # Unione dei DataFrame in uno singolo
    merged_df = pd.merge(dati_aggregati_pil, dati_aggregati_produzione, on='TIME', how='outer')
    merged_df = pd.merge(merged_df, dati_aggregati_sp, on='TIME', how='outer')
    merged_df = pd.merge(merged_df, dati_aggregati_rm, on='TIME', how='outer')

    # Rimozione delle righe con valori mancanti (NaN)
    merged_df.dropna(inplace=True)

    # Calcolo della matrice di correlazione
    numeric_columns = ['Value_pil', 'Value_produzione', 'Value_sp', 'Value_rm']
    correlation_matrix = merged_df[numeric_columns].corr()

    return correlation_matrix, merged_df

def plot_correlation_heatmap(correlation_matrix):
    # Visualizzazione della matrice di correlazione tramite heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Matrice di Correlazione tra PIL, Produzione, RLG e RM')
    plt.show()
