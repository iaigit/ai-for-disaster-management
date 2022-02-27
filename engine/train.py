from tqdm import tqdm
import torch
import os

from .train_step import train_engine
from .val_step import val_engine

def train(train_dataloaders, val_dataloaders, model, loss_fn, optim, num_epochs, log_freq=10, save_best_model=False, best_model_name='best_model.pth', last_model_name='last_model.pth'):
    """
    Train the model for a given number of epochs.
    :param train_dataloaders: A dictionary of dataloaders for training and validation.
    :param val_dataloaders: A dictionary of dataloaders for validation.
    :param model: The model to train.
    :param loss_fn: The loss function to use.
    :param optim: The optimizer to use.
    :param num_epochs: The number of epochs to train for.
    :param log_freq: The frequency with which to log training metrics.
    :return: The trained model.
    """
    best_model = None
    best_val_loss = float('inf')

    best_model_name = os.path.join('ckpt_save', best_model_name)
    last_model_name = os.path.join('ckpt_save', last_model_name)

    for epoch in range(num_epochs):
        train_loss = train_engine(train_dataloaders, model, loss_fn, optim)
        val_loss = val_engine(val_dataloaders, model, loss_fn)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_best_model:
                best_model = model
                torch.save(best_model.state_dict(), best_model_name)
                torch.save(model.state_dict(), last_model_name)

        if epoch % log_freq == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
            print('Train Loss: {:.4f}'.format(train_loss))
            print('Val Loss: {:.4f}'.format(val_loss))
            print()

    return model