# validation.py

import torch
from torch.utils.data import DataLoader
from utils.data_utils import filter_and_prepare_tensor, normalize_tensor, create_sequences
from models.lstm_model import LSTMModel 
import pandas as pd

def validate_model(model, val_dataloader, criterion, tensor_pil, tensor_prod, tensor_rlg, tensor_rm):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm, batch_target in val_dataloader:
            output = model(batch_seq_pil, batch_seq_prod, batch_seq_rlg, batch_seq_rm)
            loss = criterion(output.squeeze(), batch_target)
            val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        return avg_val_loss
