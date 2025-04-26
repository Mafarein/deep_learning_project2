import pandas as pd
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from data_loading import create_dataloader, COMMANDS
from model import SimpleSpeechCommandModel
from model_training import train_model
from transformer import AudioTransformer

# add unknown, silence percentage for data loader


def default_parameters(model_type, lr, max_epochs, seed):
    '''
    Default parameters for training
    '''
    if model_type == 'transformer':
        model = AudioTransformer(n_classes=num_classes)
    elif model_type == 'cnn': #change name later
        model = SimpleSpeechCommandModel(num_classes=num_classes)
    else:
        raise ValueError('Unknown model type')
    train_loader = create_dataloader(data_dir, batch_size=32, mode='train')
    val_loader = create_dataloader(data_dir, batch_size=32, mode='validation')
    train_losses, val_losses = train_model(model, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr,
                                           seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(
        f'saved_losses/{model_type}_constant_lr_{lr}_seed_{seed}.csv', index=False)
    torch.save(model.state_dict(), f'saved_models/{model_type}_constant_lr_{lr}_seed_{seed}.pth')

def dropout(model_type, lr, max_epochs, seed, dropout_rate=0.2):
    '''
    Dropout with 20% dropout rate
    '''
    if model_type == 'transformer':
        model = AudioTransformer(n_classes=num_classes, dropout=dropout_rate)
    elif model_type == 'cnn':  # change name later
        model = SimpleSpeechCommandModel(num_classes=num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError("Invalid model name")
    train_loader = create_dataloader(data_dir, batch_size=32, mode='train')
    val_loader = create_dataloader(data_dir, batch_size=32, mode='validation')
    train_losses, val_losses = train_model(model, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr,
                                           seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model_type}_dropout_{dropout_rate}_seed_{seed}.csv', index=False)
    torch.save(model.state_dict(), f'saved_models/{model_type}_dropout_{dropout_rate}_seed_{seed}.pth')

def transformer_experiments(lr, max_epochs, seed, d_model=128, nhead=4, num_layers=4, dropout=0):
    '''
    Transformer experiments with different parameters
    '''
    model = AudioTransformer(n_classes=num_classes, d_model=d_model, nhead=nhead, num_layers=num_layers, dropout=dropout)
    train_loader = create_dataloader(data_dir, batch_size=32, mode='train')
    val_loader = create_dataloader(data_dir, batch_size=32, mode='validation')
    train_losses, val_losses = train_model(model, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr,
                                           seed=seed)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/transformer_experiment_{d_model}_{nhead}_{num_layers}_{dropout}_seed_{seed}.csv', index=False)
    torch.save(model.state_dict(), f'saved_models/transformer_experiment_{d_model}_{nhead}_{num_layers}_{dropout}_seed_{seed}.pth')

def weight_decay(model_type, lr, max_epochs, seed, weight_decay=0.01):
    if model_type == 'transformer':
        model = AudioTransformer(n_classes=num_classes)
    elif model_type == 'cnn':  # change name later
        model = SimpleSpeechCommandModel(num_classes=num_classes)
    else:
        raise ValueError("Invalid model name")
    train_loader = create_dataloader(data_dir, batch_size=32, mode='train')
    val_loader = create_dataloader(data_dir, batch_size=32, mode='validation')
    train_losses, val_losses = train_model(model, train_loader, val_loader, max_epochs=max_epochs, learning_rate=lr,
                                           seed=seed, weight_decay=weight_decay)
    pd.DataFrame({'train_loss': train_losses, 'val_loss': val_losses}).to_csv(f'saved_losses/{model_type}_weight_decay_{weight_decay}_seed_{seed}.csv', index=False)
    torch.save(model.state_dict(), f'saved_models/{model_type}_weight_decay_{weight_decay}_seed_{seed}.pth')

def load_model(model_path, num_classes, model_type="cnn", **kwargs):
    if model_type == "cnn":
        model = SimpleSpeechCommandModel(num_classes=num_classes)
    elif model_type == "transformer":
        model = AudioTransformer(n_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
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
    seed = 1
    epochs = 50

    # default_parameters('transformer', lr=0.0001, max_epochs=epochs, seed=seed)
    #dropout('cnn', lr=0.001, max_epochs=epochs, seed=seed)


    # for i in range(1, 11):
    #     seed = i
    #     dropout('transformer', lr=0.0001, max_epochs=epochs, seed=seed, dropout_rate=0.3)
    for i in range(1, 6):
        transformer_experiments(lr=0.0001, max_epochs=epochs, seed=i, d_model=512)



    #model_path = 'saved_models/cnn_constant_lr_0.001_seed_123.pth'
    #model = load_model(model_path, num_classes=num_classes)

    #test_loader = create_dataloader(data_dir, batch_size=32, mode='testing')
    #cm = compute_confusion_matrix(model, test_loader)
    #plot_confusion_matrix(cm, class_names=COMMANDS)
