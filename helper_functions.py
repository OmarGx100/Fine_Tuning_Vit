# import needed dependencies
import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path
import os
import pathlib
import matplotlib.pyplot as plt
import PIL
import numpy as np

from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict


device = 'cuda' if torch.cuda.is_available() else 'cpu'

device

# accuracy function
def accuracy_fn(y_pred, y_true):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

# Create train and test loop functions

#setting device agnostic code 
# device = 'cpu'

def train_step(model : torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fnc : torch.nn.Module,
               optimzer : torch.optim.Optimizer,
               acc_fnc,
               device : torch.device = device):
    """Performce a training step on the dataloader"""
    train_loss = 0
    train_acc = 0
    model.to(device)
    model.train()
    for X, y in data_loader:
        # send the data to the target device        
        X, y = X.to(device), y.to(device)
        # Do forward path
        y_preds = model(X)
        #calculate the training loss
        loss = loss_fnc(y_preds, y)
        train_loss += loss.item()
        #optimzer zero grad
        optimzer.zero_grad()
        # loss backward
        train_acc += acc_fnc(y_preds.argmax(dim = 1), y)
        #Loss backward
        loss.backward()
        #optimzer step
        optimzer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    #return train loss and train acc
    return train_loss, train_acc


def test_step(model : torch.nn.Module,
               data_loader : torch.utils.data.DataLoader,
               loss_fnc : torch.nn.Module,
               acc_fnc,
               device : torch.device = device):
    """Performce a testing step on the dataloader"""

    model.to(device)
    test_loss = 0
    test_acc = 0
    model.eval()
    for X, y in data_loader:
        
        # Sending the data to target device
        X, y = X.to(device), y.to(device)

        #Forward path
        with torch.inference_mode():
            y_preds = model(X)
        
        # Calculate the loss
        loss = loss_fnc(y_preds, y)
        test_loss += loss.item()

        # Calculate test accuracy
        test_acc += acc_fnc(y_preds.argmax(dim = 1), y)
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)

    return test_loss, test_acc
    
def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTrain time on {device}: {total_time:.3f} seconds")
    return total_time

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train(model: torch.nn.Module,
          train_data_loader:torch.utils.data.DataLoader,
          test_data_loader : torch.utils.data.DataLoader,
          optimzer : torch.optim.Optimizer,
          loss_fnc : torch.nn.Module,
          acc_fnc,
          calculate_train_time : bool = True,
          seed : int = None,
          device : torch.device = device,
          epochs : int = 10,
          early_stop : bool = False,
          patience : int = 1,
          min_delta : int = 0):
    
    if early_stop:
        early_stopper = EarlyStopper(patience=patience,
                                     min_delta = min_delta)
    else :
        early_stopper = None


    if calculate_train_time:
        train_time_start = timer()
    

    if seed : 
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    results = {"train_loss_epoch" : [],
               "train_acc_epoch" : [],
               "test_loss_epoch" : [],
               "test_acc_epoch" : [],
               "Estimated_train_time" : None
               }
    
    

    model.to(device)
    for epoch in tqdm(range(epochs)):

        #printing epoch number
        print(f"Epoch: {epoch}\n --------")

        # One training step per epoch
        train_loss, train_acc = train_step(model = model,
                   data_loader= train_data_loader,
                   loss_fnc=loss_fnc,
                   optimzer = optimzer,
                   acc_fnc = acc_fnc,
                   device = device)
        results['train_loss_epoch'].append(train_loss), results['train_acc_epoch'].append(train_acc)
        # One testing step per epoch

        test_loss , test_acc = test_step(model = model,
                  data_loader = test_data_loader,
                  loss_fnc = loss_fnc,
                  acc_fnc = acc_fnc,
                  device = device)
        
        results['test_loss_epoch'].append(test_loss), results['test_acc_epoch'].append(test_acc)
        #print what is happening 
        print(f"Train Loss : {train_loss:.4f} | Train Accuravy : {train_acc:.2f}")
        print(f"Test Loss : {test_loss:.4f} | Test Accuracy : {test_acc:.2f}")

        #early stopping
        if early_stopper.early_stop(test_loss):
            print(f"we are in epoch {epoch}")
            break
    
    train_time_end = timer()


    if calculate_train_time:
        results['Estimated_train_time'] = print_train_time(start = train_time_start, end = train_time_end)
        print_train_time(start = train_time_start, end = train_time_end)
    return results