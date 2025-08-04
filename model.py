import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error

def evaluate(model, loader, quantiles):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    criterion = [PinballLoss(tau) for tau in quantiles]

    with torch.no_grad():
        for xb, yb in loader:
            preds = model(xb)
            loss = 0
            for i in range(len(quantiles)):
                loss += criterion[i](preds[:, i], yb)
            loss /= len(quantiles)
            total_loss += loss.item()

            # For extra metrics: use q=0.5 (index 1)
            all_preds.append(preds[:, 1].cpu())
            all_targets.append(yb.cpu())

    model.train()
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    return total_loss / len(loader), r2, mae


class PinballLoss(torch.nn.Module):
    def __init__(self, quantile):
        super().__init__()
        self.quantile = quantile

    def forward(self, preds, target):
        errors = target - preds
        return torch.mean(torch.max(self.quantile * errors, (self.quantile - 1) * errors))

class QuantileLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, num_layers=2, quantiles=[0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = quantiles
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, len(quantiles))  # Output = one value per quantile

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)  # shape: (batch_size, num_quantiles)
