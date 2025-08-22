import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights


"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Max_Pool(nn.Module):
    def __init__(self, n_classes=2, feat_type='uni'):
        super().__init__()
        if feat_type == 'uni':
            self.size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            self.size = [1536, 768, 384]

        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(self.size[1], n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, coords= None):
        "dimension reduction"
        h = self.feature(h)  # Nxsize[1]       
        
        "pool"
        M, idx = torch.max(h, dim= 0, keepdim= True) # 1xsize[1]

        "attention weights"
        A = torch.zeros(h.size()[0]).to(h.device)
        A[idx] = 1

        "classify instances"
        logits = self.classifiers(M)

        "output"
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)

        results_dict = {'attention': A}  
        return logits, Y_prob, Y_hat, A, results_dict



"""
args:
    size_arg: config for network size
    dropout: whether to use dropout
    k_sample: number of positive/neg patches to sample for instance-level training
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Mean_Pool(nn.Module):
    def __init__(self, n_classes=2, feat_type='uni'):
        super().__init__()
        if feat_type == 'uni':
            self.size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            self.size = [1536, 768, 384]

        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
            
        self.feature = nn.Sequential(*fc)
        self.classifiers = nn.Linear(self.size[1] + 30, n_classes)
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature= self.feature.to(device)
        self.classifiers = self.classifiers.to(device)

    def forward(self, h, coords= None, clinic= None):
        "dimension reduction"
        h = self.feature(h)  # Nxsize[1]       
        
        "pool"
        M = torch.mean(h, dim= 0, keepdim= True) # 1xsize[1]
        # clinical feature 
        M = torch.cat([M, clinic], dim=1)

        "classify instances"      
        logits = self.classifiers(M)

        "output"
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        Y_prob = F.softmax(logits, dim = 1)
        
        return logits, Y_prob, Y_hat, {}, {}
