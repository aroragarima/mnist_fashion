import matplotlib.pyplot as plt

def plot_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = list(range(1,len(acc)+1))
    
    plt.plot(epochs, loss, 'g--', label="Training Loss")
    plt.plot(epochs, val_loss, 'r--', label="Validation Loss")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Negative Log Likelihood/Loss')

    return plt