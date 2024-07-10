import torch.nn as nn
import torch
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size_pil, input_size_prod, input_size_rlg, input_size_rm, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        
        # Inizializzazione dei parametri del modello
        self.hidden_size = hidden_size  # Dimensione dello strato nascosto
        self.num_layers = num_layers    # Numero di strati LSTM
       
        # Definizione dei layer LSTM per ogni tipo di input
        self.lstm_pil = nn.LSTM(input_size_pil, hidden_size, num_layers, batch_first=True)
        self.lstm_prod = nn.LSTM(input_size_prod, hidden_size, num_layers, batch_first=True)
        self.lstm_rlg = nn.LSTM(input_size_rlg, hidden_size, num_layers, batch_first=True)
        self.lstm_rm = nn.LSTM(input_size_rm, hidden_size, num_layers, batch_first=True)
        
        # Definizione del layer completamente connesso per l'output finale
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_pil, x_prod, x_rlg, x_rm):
        # Inizializzazione degli stati nascosti e delle memorie LSTM
        h0 = torch.zeros(self.num_layers, x_pil.size(0), self.hidden_size).to(x_pil.device)
        c0 = torch.zeros(self.num_layers, x_pil.size(0), self.hidden_size).to(x_pil.device)
        
        # Passaggio attraverso i layer LSTM per ogni tipo di input
        out_pil, _ = self.lstm_pil(x_pil, (h0, c0))
        out_prod, _ = self.lstm_prod(x_prod, (h0, c0))
        out_rlg, _ = self.lstm_rlg(x_rlg, (h0, c0))
        out_rm, _ = self.lstm_rm(x_rm, (h0, c0))
        
        # Combinazione degli output LSTM
        out_combined = out_pil[:, -1, :] + out_prod[:, -1, :] + out_rlg[:, -1, :] + out_rm[:, -1, :]
        
        # Passaggio attraverso il layer completamente connesso per ottenere l'output finale
        out = self.fc(out_combined)
        return out

