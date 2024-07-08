from torch.utils.data import DataLoader, TensorDataset
import torch

def create_dataloader(sequences, targets, batch_size, shuffle=False):
    # Creazione di un oggetto TensorDataset con le sequenze e i target
    dataset = TensorDataset(
        torch.stack([seq[0] for seq in sequences]),    # Sequenze per il primo tipo di input (PIL)
        torch.stack([seq[1] for seq in sequences]),    # Sequenze per il secondo tipo di input (Produzione)
        torch.stack([seq[2] for seq in sequences]),    # Sequenze per il terzo tipo di input (RLG)
        torch.stack([seq[3] for seq in sequences]),    # Sequenze per il quarto tipo di input (RM)
        targets  # Target associati alle sequenze
    )
    
    # Creazione di un DataLoader per gestire il batch delle sequenze durante l'addestramento
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
