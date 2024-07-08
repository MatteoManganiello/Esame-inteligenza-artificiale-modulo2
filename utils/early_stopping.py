
import torch

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.salva_checkpoint(val_loss, model)
        elif val_loss > self.best_score:
            self.counter += 1
            if self.verbose:
                print(f'Contatore di EarlyStopping: {self.counter} su {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.salva_checkpoint(val_loss, model)
            self.counter = 0

    def salva_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Diminuzione della loss di validazione ({self.best_score:.6f} --> {val_loss:.6f}). Salvataggio del modello...')
        torch.save(model.state_dict(), 'miglior_modello.pth')
        self.best_score = val_loss
