
import torch
import torch.nn as nn
import torch.optim as optim
import pkbar
import time

from torch.autograd import Variable
from utils.CustomLoss import PWSLoss, MBWHLoss, QuantLoss


def Train(model, data_loader, eval_data_loader, batch_size, optimizer, epoch, device, log):

    # Set mode for model in train
    model.train() 
    epochs = 0
    for batch_idx, samples in enumerate(data_loader):
        
        # Initialize parameters of optimizer
        optimizer.zero_grad()

        # Load dataset and set in device
        img, _ = samples
        img = img.to(device) 
        
        # Forward propagate
        label_pred, hash_pred = model(img)
        
        # Calculate loss with custom loss method
        l1_loss = PWSLoss(label_pred, hash_pred)
        l2_loss = MBWHLoss(label_pred, hash_pred, device)
        l3_loss = QuantLoss(label_pred, hash_pred)
        weighted_loss = l1_loss + 10*l2_loss + l3_loss
        
        # Record logs for monitoring
        log.add_scalar("log/l1_loss", l1_loss)
        log.add_scalar("log/l2_loss", l2_loss)
        log.add_scalar("log/l3_loss", l3_loss)
        log.add_scalar("log/all_loss", weighted_loss)
        
        # Pring loss values
        print_loss(l1_loss, l2_loss, l3_loss, weighted_loss, batch_idx, epoch)

        # Back propagate
        weighted_loss.backward()
        optimizer.step()
        evaluate(model, eval_data_loader, device, batch_idx, epoch)

    torch.save(model.state_dict(), parser.save_path+'model{epochs}.pt'.format())
    torch.save(model, parser.save_path+'model_dict{epochs}.pt')
    epochs += 1

def evaluate(model, eval_data_loader, device, batch_idx, epoch):
    
    # Set mode for model in evaluation
    model.eval()
    
    for batch_idx, samples in enumerate(eval_data_loader):
        img, _ = samples
        img = img.to(device)

        # Forward propagate
        label_pred, hash_pred = model(img)

        #Calculate loss with custom loss method
        l1_loss = PWSLoss(label_pred, hash_pred)
        l2_loss = MBWHLoss(label_pred, hash_pred, device)
        l3_loss = QuantLoss(label_pred, hash_pred)
        weighted_loss = l1_loss + 10*l2_loss + l3_loss
        
        print("Eval result")
        print_loss(l1_loss, l2_loss, l3_loss, weighted_loss, batch_idx, epoch)
        print("========================================================")
def print_loss(l1_loss, l2_loss, l3_loss, total_loss, batch_idx, epoch):
    print("Epoch : {}".format(epoch))
    print("Batch : {}".format(batch_idx))
    print("L1 Loss {:.4f},  L2 Loss {:.4f},  L3 Loss {:.4f},  Total Loss {:.4f}".format(l1_loss, 
        l2_loss, l3_loss, total_loss))
    print("\n")
