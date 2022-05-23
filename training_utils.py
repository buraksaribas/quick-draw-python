import random
import numpy as np
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

def train_epoch(model, train_loader, optimizer, device):
    model.train()
    for x, y_true in tqdm(train_loader, leave=False):
        # Move to GPU, in case there is one
        x, y_true = x.to(device), y_true.to(device)
        
        # Compute logits
        y_lgts = model(x)
        
        # Compute the loss
        loss = F.cross_entropy(y_lgts, y_true)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_epoch(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        losses, accs = [], []

        for x, y_true in tqdm(data_loader, leave=False):
            # Move to GPU, in case there is one
            x, y_true = x.to(device), y_true.to(device)

            # Compute logits
            y_lgts = model(x)

            # Compute scores
            y_prob = F.softmax(y_lgts, dim=1)

            # Get the classes
            y_pred = torch.argmax(y_prob, dim=1)

            # Compute the loss
            loss = F.cross_entropy(y_lgts, y_true)

            # Compute accuracy
            accuracy = (y_true == y_pred).type(torch.float32).mean()

            # Save the current loss and accuracy
            losses.append(loss.item())
            accs.append(accuracy.item())

        # Compute the mean
        loss = np.mean(losses) * 100
        accuracy = np.mean(accs) * 100

        return loss, accuracy

def save_checkpoint(model, optimizer, epoch, loss, path):
    if path:
        torch.save(
            {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(), 
             'loss': loss,
             }, path)

def train(model, train_loader, validation_loader, device, lr=0.001, epochs=20, patience=5, 
          writer=None, checkpoint_path=None):

    # https://clay-atlas.com/us/blog/2021/08/25/pytorch-en-early-stopping/
    last_loss = np.inf
    early_stop = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in trange(epochs):
        # Train a single epoch
        train_epoch(model, train_loader, optimizer, device)

        # Evaluate in training
        train_loss, train_acc = eval_epoch(model, train_loader, device)
        # Evaluate in validation
        val_loss, val_acc = eval_epoch(model, validation_loader, device)

        if writer:
            writer.add_scalar(tag='Loss/train', scalar_value=train_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/train', scalar_value=train_acc, global_step=epoch)

            writer.add_scalar(tag='Loss/validation', scalar_value=val_loss, global_step=epoch)
            writer.add_scalar(tag='Accuracy/validation', scalar_value=val_acc, global_step=epoch)

        # Early stopping
        current_loss = val_loss
        if current_loss > last_loss:
            early_stop += 1
            if early_stop > patience:
                print('Early stopping!')
                return # Stop training
        else:
            early_stop = 0
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)

        last_loss = current_loss

def create_model():
    model = models.mobilenet_v2()
    
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features=1280, out_features=345)
    )

    return model