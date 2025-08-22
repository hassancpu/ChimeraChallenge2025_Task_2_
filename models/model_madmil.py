
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

"""
Attention Network with Sigmoid Gating (3 fc layers)
args:
    L: input feature dimension
    D: hidden layer dimension
    dropout: whether to use dropout (p = 0.25)
    n_classes: number of classes 
"""
class Attn_Net_Gated(nn.Module):
    def __init__(self, L = 1024, D = 256, dropout = False, n_classes = 1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        return A, x
    

"""
Multihead ABMIL model with seperate attention modules.

Args:
    gate (bool): whether to use gated attention network
    size_arg (str): config for network size
    dropout (bool): whether to use dropout
    n_classes (int): number of classes
    n (int): number of attention heads
    head_size (str): size of each head
"""
            
class MADMIL(nn.Module):
    def __init__(self, n_classes=2, n=2, feat_type='uni'):
        super().__init__()
        if feat_type == 'uni':
            self.size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            self.size = [1536, 768, 384]

        # number of attention heads
        self.n_heads = n
        
        if self.size[1] % self.n_heads != 0:
           print("The feature dim should be divisible by num_heads!! Do't worry, we will fix it for you.")
           self.size[1] = math.ceil(self.size[1] / self.n_heads) * self.n_heads

        # size of each head              
        self.step = self.size[1] // self.n_heads  
        self.dim = self.step // 2
   
        
        fc = [nn.Linear(self.size[0], self.size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        att_net = [Attn_Net_Gated(L=self.step, D=self.dim, dropout=True, n_classes=1) for ii in range(self.n_heads)]

        self.net_general = nn.Sequential(*fc)
        self.attention_net =  nn.ModuleList(att_net)
        self.classifiers = nn.Linear(self.size[1] + 30, n_classes) 
        initialize_weights(self)

    def relocate(self):
        """
        Relocates the model to GPU if available, else to CPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net_general = self.net_general.to(device)
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        
    def forward(self, h, coords= None, clinic= None, attention_only=False):
        """
        Forward pass of the model.

        Args:
            h (torch.Tensor): Input tensor
            attention_only (bool): Whether to return only attention weights

        Returns:
            tuple: Tuple containing logits, predicted probabilities, predicted labels, attention weights, and attention weights before softmax
        """
   
        h = self.net_general(h)
        N, _ = h.shape

        # Multihead Input
        h = h.reshape(N, self.n_heads, self.step) # Nxheadsxstep
        A = torch.stack([self.attention_net[nn](h[:,nn,:])[0] for nn in range(self.n_heads)], dim=1)

        A = torch.transpose(A, 2, 0)  # KxheadsxN
        if attention_only:
            return A
        A_raw = A
           
        A = F.softmax(A, dim=-1)  # softmax over N     
        
        # Multihead Output
        M = torch.einsum('ijk,kjl->ijl', A, h)
        M = M.view(1, self.size[1])
        # clinical feature 
        M = torch.cat([M, clinic], dim=1)

        # Singlehead Classification
        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1) 

        results_dict = {'attention': A_raw} 
        return logits, Y_prob, Y_hat, A_raw, results_dict 
    

