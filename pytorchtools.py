import numpy as np
import torch
import pickle
import os 

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
#         self.checkpointfilename = checkpointfilename
        
    def __call__(self, val_loss, model, modelname, parentfolder):
        
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model,modelname, parentfolder)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter,self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, modelname, parentfolder)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, modelname,parentfolder):
        
#         parentfolder = r"/mnt/ccipd/home/axh672/testretest/Tesla/"
        parentfolder = r"Data/"


        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({} --> {}).  Saving model ...'.format(self.val_loss_min,val_loss))

            
        if not os.path.exists("{}modelcheckpoints/{}".format(parentfolder,modelname)):
            os.makedirs("{}modelcheckpoints/{}".format(parentfolder,modelname))
            
        torch.save(model.state_dict(), "{}modelcheckpoints/{}/checkpoint.pt".format(parentfolder,modelname))
            
        self.val_loss_min = val_loss