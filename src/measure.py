import torch


def mnist_loss(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets == 1, 1 - predictions, predictions).mean()


def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, valid_dl):
    model.eval()
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    model.train()
    return round(torch.stack(accs).mean().item(), 4)

def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()


def validate_epoch(model, valid_dl):
    accs = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accs).mean().item(), 4)
