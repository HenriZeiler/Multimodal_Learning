import torch
from torchmetrics.classification import MulticlassF1Score
import torch.nn as nn



def train_on_batch_and_get_loss(model, batch):
    """Return a batch-level loss function for any dual-input (rgb, xyza) model."""
    rgb, xyza, labels = batch
    print(len(rgb))
    print(len(xyza))
    prediction = model(rgb, xyza)
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(prediction, labels), prediction

def run_training(model, optimizer, epochs, train_loader, val_loader, device, batch_runner_and_loss_fn = train_on_batch_and_get_loss, calculate_f1_score = True):
    """Train model, print per-epoch losses, return (train_losses, val_losses)."""
    train_losses, val_losses = [], []
    best_f1_score = None
    if calculate_f1_score: 
        f1_scores = MulticlassF1Score(num_classes=2, average="macro").to(device)
        best_f1_score = 0.0
    for epoch in range(epochs):
        model.train()
        t_loss = 0
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss, _ = batch_runner_and_loss_fn(model, batch)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
        t_loss /= (step + 1)
        train_losses.append(t_loss)
        model.eval()
        v_loss_acc = 0
        if calculate_f1_score: 
            f1_scores.reset()
        with torch.no_grad():
            total_correct = 0
            for step, batch in enumerate(val_loader):
                v_loss, pred = batch_runner_and_loss_fn(model, batch)
                _, _, labels = batch
                pred_correct = (torch.sigmoid(pred) > 0.5).long()
                total_correct += (pred_correct == labels.long()).sum().item()
                if calculate_f1_score: 
                    f1_scores.update(pred_correct, labels)
                v_loss_acc += v_loss.item()
        val_accuracy = total_correct / len(val_loader.dataset)
        v_loss_acc /= (step + 1)
        val_losses.append(v_loss_acc)
        if calculate_f1_score:
            best_f1_score = max(best_f1_score, f1_scores.compute().item())
            print(best_f1_score)
        print(f"Epoch {epoch:3d} | Train: {t_loss:.4f}  Val: {v_loss_acc:.4f}")
    return train_losses, val_losses, best_f1_score, val_accuracy #last epocs validation acc
