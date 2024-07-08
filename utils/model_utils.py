import torch
import numpy as np

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Modello salvato con successo in {filepath}")

def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()
    print(f"Modello caricato con successo da {filepath}")