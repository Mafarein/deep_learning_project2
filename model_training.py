import torch
import torch.nn as nn
import torch.optim as optim
from early_stopping import EarlyStopper
from tqdm import tqdm
import random
import numpy as np


def train_model(model, train_loader, val_loader, max_epochs, learning_rate, lr_scheduling=False,
                seed=123):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    if lr_scheduling:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    train_losses = []
    val_losses = []

    early_stopper = EarlyStopper(patience=10)

    tqdm_epochs = tqdm(range(max_epochs), desc='Epochs', leave=True)

    for epoch in tqdm_epochs:
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_train_loss = running_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

        average_val_loss = running_val_loss / len(val_loader)

        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)

        tqdm_epochs.set_postfix(train_loss=average_train_loss, val_loss=average_val_loss)

        if early_stopper.early_stop(average_val_loss):
            print(f'Early stopping at epoch {epoch}')
            break

        if lr_scheduling:
            scheduler.step(running_loss)

    return train_losses, val_losses