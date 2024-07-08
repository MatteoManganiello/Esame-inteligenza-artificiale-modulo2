import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from utils.data_utils import load_config, filter_and_prepare_tensor, normalize_tensor, create_sequences
from data.dataloader import create_dataloader
from models.lstm_model import LSTMModel 
from utils.data_processing import compute_correlation_matrix
from utils.early_stopping import EarlyStopping
from validation import validate_model  
import matplotlib.pyplot as plt
from predictions_and_correlation import plot_correlation_matrix
from predictions_and_correlation import plot_correlation_matrix



def main():
    # Carica la configurazione da file JSON
    config = load_config('config/config.json', 'config/config_schema.json')

    # Carica il dataset da file CSV
    try:
        file_path = config['data']['file_path']
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore durante il caricamento del dataset: {e}")
        return

    # Definizione dei filtri per il dataset
    filters_pil = {"tipo_dato": ["B1GQ_B_W2_S1_X2"], "edizione": ["Set-2023"]}
    filters_prod = {"tipo_dato": ["P1_C_W2_S1"], "edizione": ["Set-2023"], "valutazione": ["prezzi correnti"]}
    filters_rlg = {"tipo_dato": ["P3_D_W0_S1"], "edizione": ["Set-2023"], "valutazione": ["prezzi correnti"]}
    filters_rm = {"tipo_dato": ["B2A3G_B_W2_S1"], "edizione": ["Set-2023"]}
    columns_to_remove = ['Seleziona periodo', 'Flag Codes', 'Flags']

    # Filtraggio e preparazione dei tensori per ciascun filtro
    try:
        tensor_pil = filter_and_prepare_tensor(df, filters_pil, columns_to_remove)
        tensor_prod = filter_and_prepare_tensor(df, filters_prod, columns_to_remove)
        tensor_rlg = filter_and_prepare_tensor(df, filters_rlg, columns_to_remove)
        tensor_rm = filter_and_prepare_tensor(df, filters_rm, columns_to_remove)

        # Normalizzazione dei tensori filtrati
        tensor_pil_normalized = normalize_tensor(tensor_pil)
        tensor_prod_normalized = normalize_tensor(tensor_prod)
        tensor_rlg_normalized = normalize_tensor(tensor_rlg)
        tensor_rm_normalized = normalize_tensor(tensor_rm)
    except Exception as e:
        print(f"Errore durante il filtraggio e la preparazione dei tensori: {e}")
        return

    # Creazione delle sequenze e dei target per l'addestramento del modello
    try:
        seq_length = config['data']['seq_length']
        sequences, targets = create_sequences(tensor_pil_normalized, tensor_prod_normalized, tensor_rlg_normalized, tensor_rm_normalized, seq_length=seq_length)
    except Exception as e:
        print(f"Errore durante la creazione delle sequenze e dei target: {e}")
        return

    # Creazione del DataLoader per gestire i dati in batch durante l'addestramento
    try:
        batch_size = config['training']['batch_size']
        train_dataloader = create_dataloader(sequences, targets, batch_size)
    except Exception as e:
        print(f"Errore durante la creazione del DataLoader: {e}")
        return

    # Parametri del modello
    try:
        input_size_pil = tensor_pil_normalized.shape[1]
        input_size_prod = tensor_prod_normalized.shape[1]
        input_size_rlg = tensor_rlg_normalized.shape[1]
        input_size_rm = tensor_rm_normalized.shape[1]
        hidden_size = config['model']['hidden_size']
        num_layers = config['model']['num_layers']
        output_size = config['model']['output_size']

        # Creazione del modello
        model = LSTMModel(input_size_pil, input_size_prod, input_size_rlg, input_size_rm, hidden_size, num_layers, output_size)
    except Exception as e:
        print(f"Errore durante la creazione del modello LSTM: {e}")
        return

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # Inizializza EarlyStopping
    early_stopping = EarlyStopping(patience=config['training']['early_stopping_patience'], verbose=True)

    # Addestramento del modello
    try:
        num_epochs = config['training']['num_epochs']
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            for batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm, batch_target in train_dataloader:
                optimizer.zero_grad()
                output = model(batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm)
                loss = criterion(output.squeeze(), batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(train_dataloader)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

            # Validazione del modello se abilitata
            if config.get('validation', {}).get('enable_validation', False):
                if (epoch + 1) % config['validation']['validate_every'] == 0:
                    val_dataloader = create_dataloader(sequences, targets, batch_size, shuffle=False)
                    avg_val_loss = validate_model(model, val_dataloader, criterion)
                    print(f'Validation Loss after {epoch + 1} epochs: {avg_val_loss:.4f}')

                    # Applica Early Stopping
                    early_stopping(avg_val_loss, model)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    # Salva il modello se la loss di validazione è migliorata
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), 'best_model.pth')

        # Salva il modello se la validazione non è abilitata
        if not config.get('validation', {}).get('enable_validation', False):
            torch.save(model.state_dict(), 'best_model.pth')

        # Calcola la loss di test
        if config.get('test', {}).get('enable_test', False):
            test_dataloader = create_dataloader(sequences, targets, batch_size, shuffle=False)
            test_loss = validate_model(model, test_dataloader, criterion)
            print(f'Test Loss: {test_loss:.4f}')

    except Exception as e:
        print(f"Errore durante l'addestramento del modello: {e}")
        return

    # Carica il modello migliore se esiste, altrimenti addestralo
    if os.path.exists('best_model.pth'):
        try:
            model.load_state_dict(torch.load('best_model.pth'))
            model.eval()

            # Previsioni future
            num_predictions = config['prediction']['num_predictions']
            predicted_values = []
            with torch.no_grad():
                last_sequence_pil = sequences[-1][0].unsqueeze(0)  # Ultima sequenza PIL
                last_sequence_prod = sequences[-1][1].unsqueeze(0)  # Ultima sequenza Produzione
                last_sequence_rlg = sequences[-1][2].unsqueeze(0)  # Ultima sequenza RLG
                last_sequence_rm = sequences[-1][3].unsqueeze(0)  # Ultima sequenza RM
                for _ in range(num_predictions):
                    predicted_value_normalized = model(last_sequence_pil, last_sequence_prod, last_sequence_rlg, last_sequence_rm).item()
                    # Denormalizzazione del valore predetto
                    min_pil = tensor_pil.min(dim=0, keepdim=True).values[0, 1]
                    max_pil = tensor_pil.max(dim=0, keepdim=True).values[0, 1]
                    predicted_value = predicted_value_normalized * (max_pil - min_pil) + min_pil
                    predicted_values.append(predicted_value)

                    # Aggiornamento delle sequenze con la nuova previsione
                    new_sequence_pil = torch.cat((last_sequence_pil[:, 1:, :], torch.tensor([[[tensor_pil[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
                    new_sequence_prod = torch.cat((last_sequence_prod[:, 1:, :], torch.tensor([[[tensor_prod[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
                    new_sequence_rlg = torch.cat((last_sequence_rlg[:, 1:, :], torch.tensor([[[tensor_rlg[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)
                    new_sequence_rm = torch.cat((last_sequence_rm[:, 1:, :], torch.tensor([[[tensor_rm[-1, 0] + 1, predicted_value_normalized]]], dtype=torch.float32)), dim=1)

                    last_sequence_pil = new_sequence_pil
                    last_sequence_prod = new_sequence_prod
                    last_sequence_rlg = new_sequence_rlg
                    last_sequence_rm = new_sequence_rm

                    
            # Stampa delle previsioni future
            for i, value in enumerate(predicted_values, 1):
                print(f'Valore predetto per l\'anno {int(tensor_pil[-1, 0].item()) + i}: {value:.2f}') 

            predicted_years = range(int(tensor_pil[-1, 0].item()) + 1, int(tensor_pil[-1, 0].item()) + num_predictions + 1)
            
            # Grafico delle previsioni
            plt.figure(figsize=(10, 6))
            plt.plot(predicted_years, predicted_value, marker='o', linestyle='-', color='b', label='Valori Previsti')
            plt.title('')
            plt.xlabel('Anni')
            plt.ylabel('Valore')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()


        except Exception as e:
            print(f"Errore durante le previsioni future: {e}")
            return
    else:
        print("Il file 'best_model.pth' non esiste. Assicurati di aver completato l'addestramento del modello.")
        return
         
    # Calcolo della matrice di correlazione
    try:
        plot_correlation_matrix(df)
    except Exception as e:
        print(f"Errore durante il calcolo della matrice di correlazione: {e}")
        return

if __name__ == "__main__":
    main()
