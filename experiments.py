import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from data_loading import create_dataloader, COMMANDS
from model import SimpleSpeechCommandModel
from model_training import train_model


def default_parameters(lr, max_epochs, seed):
    '''
    Default parameters for training
    '''
    cnn = SimpleSpeechCommandModel(num_classes=num_classes)
    train_loader = create_dataloader(data_dir, batch_size=32, mode='train')
    val_loader = create_dataloader(data_dir, batch_size=32, mode='validation')
    train_losses, val_losses = train_model(cnn, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr,
                                           seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(
        f'test.csv', index=False)
    torch.save(cnn.state_dict(), f'test_{seed}.pth')


def load_model(model_path, num_classes):
    model = SimpleSpeechCommandModel(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def compute_confusion_matrix(model, data_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    return cm

def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    data_dir = "data/train"
    num_classes = len(COMMANDS) + 2

    #default_parameters(lr=0.001, max_epochs=10, seed=123)

    model_path = 'test_123.pth'
    model = load_model(model_path, num_classes=num_classes)

    test_loader = create_dataloader(data_dir, batch_size=32, mode='testing')
    cm = compute_confusion_matrix(model, test_loader)
    plot_confusion_matrix(cm, class_names=COMMANDS)
