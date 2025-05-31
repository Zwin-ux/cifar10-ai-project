import matplotlib.pyplot as plt

def plot_train_val_curves(train_losses, val_losses, train_accs, val_accs, title_suffix=''):
    '''Plots training and validation loss and accuracy curves.'''
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    if val_losses:
        plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Loss Curves {title_suffix}')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    if val_accs:
        plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy Curves {title_suffix}')

    plt.tight_layout()
    plt.show()
