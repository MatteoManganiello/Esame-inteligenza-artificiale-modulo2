import pandas as pd
import torch
import json
from jsonschema import validate, ValidationError

def load_config(config_path, schema_path):
    # Carica il file di configurazione JSON
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # Carica lo schema JSON per la validazione
    with open(schema_path, 'r') as schema_file:
        schema = json.load(schema_file)
    
    try:
        # Valida il file di configurazione rispetto allo schema
        validate(instance=config, schema=schema)
    except ValidationError as e:
        # Se la validazione fallisce, solleva un'eccezione con un messaggio dettagliato
        raise ValueError(f"Configurazione non valida: {e.message}")
    
    return config

def load_and_preprocess_data(file_path):
    # Carica i dati da un file CSV utilizzando pandas
    df = pd.read_csv(file_path)
    
    # Filtraggio e pre-processamento dei dati
    # (I dettagli specifici del pre-processamento possono variare a seconda dei requisiti specifici)
    
    return df

def filter_and_prepare_tensor(df, filters, columns_to_remove):
    # Applica i filtri specificati al DataFrame
    dati_filtrati = df[(df['TIPO_DATO_PIL_SEC2010'].isin(filters['tipo_dato'])) &
                       (df['Edizione'].isin(filters['edizione']))]
    if 'valutazione' in filters:
        dati_filtrati = dati_filtrati[dati_filtrati['Valutazione'].isin(filters['valutazione'])]

    # Rimuove colonne non necessarie e gestisce i valori mancanti
    dati_completi = dati_filtrati.drop(columns=columns_to_remove, axis=1).fillna(0)
    
    # Seleziona solo colonne di tipo numerico (float e int)
    dati_completi = dati_completi.select_dtypes(include=[float, int])
    
    # Converte i dati in un array NumPy e poi in un tensore PyTorch
    array_dati = dati_completi.to_numpy()
    tensor_dati = torch.tensor(array_dati, dtype=torch.float32)
    
    return tensor_dati

def normalize_tensor(tensor):
    # Normalizza il tensore portandolo in un range [0, 1] rispetto ai suoi valori minimi e massimi
    tensor_min = tensor.min(dim=0, keepdim=True).values
    tensor_max = tensor.max(dim=0, keepdim=True).values
    return (tensor - tensor_min) / (tensor_max - tensor_min)

def create_sequences(*tensors, seq_length):
    sequences = []
    targets = []
    
    # Creazione delle sequenze di lunghezza `seq_length` e dei corrispondenti target
    for i in range(len(tensors[0]) - seq_length):
        seqs = [tensor[i:i+seq_length, :] for tensor in tensors]
        target = tensors[0][i+seq_length, 1]  # Prendiamo solo il PIL come target
        sequences.append(seqs)
        targets.append(target)
    
    # Converte le liste di sequenze in un tensore PyTorch per le sequenze e un tensore per i target
    return sequences, torch.tensor(targets)
