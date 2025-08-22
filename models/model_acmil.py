import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights

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
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]
        
        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
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
args:
    n_classes: number of classes
    n_token: number of attention heads
    n_masked_patch: number of masked patches (0 for no masking)
    mask_drop: dropout rate for masked patches
"""

class ACMIL(nn.Module):
    def __init__(self, n_classes=2, n_token=1, n_masked_patch=0, mask_drop=0.6, feat_type= 'imagenet'):
        super(ACMIL, self).__init__()
        if feat_type == 'imagenet':
            size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            size = [1536, 768, 384]

        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        
        attention_net = Attn_Net_Gated(L = size[1], D = size[2], dropout = True, n_classes = n_token)
            
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        
        self.classifiers = nn.ModuleList()
        for _ in range(n_token):
            self.classifiers.append(nn.Linear(size[1], n_classes))
            
        self.bag_classifier = nn.Linear(size[1], n_classes)
        self.n_classes = n_classes
        self.n_masked_patch = n_masked_patch
        self.n_token = n_token
        self.mask_drop = mask_drop
        initialize_weights(self)

    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.bag_classifier = self.bag_classifier.to(device)
        
    def forward(self, h, coords= None, attention_only=False, use_attention_mask=False):
        A, h = self.attention_net(h)  # NxK  
              
        A = torch.transpose(A, 1, 0)  # KxN
        
        if self.n_masked_patch > 0 and use_attention_mask:
            # Get the indices of the top-k largest values
            k, n = A.shape
            n_masked_patch = min(self.n_masked_patch, n)
            _, indices = torch.topk(A, n_masked_patch, dim=-1)
            rand_selected = torch.argsort(torch.rand(*indices.shape), dim=-1)[:,:int(n_masked_patch * self.mask_drop)]
            masked_indices = indices[torch.arange(indices.shape[0]).unsqueeze(-1), rand_selected]
            random_mask = torch.ones(k, n).to(A.device)
            random_mask.scatter_(-1, masked_indices, 0)
            A = A.masked_fill(random_mask == 0, -1e9)
        
        if attention_only:
            return A
        A_raw = A
            
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, h) 
        outputs = []
        for i, head in enumerate(self.classifiers):
            outputs.append(head(M[i]))
        
        # compute the average attention for the bag
        bag_A = F.softmax(A_raw, dim=1).mean(0, keepdim=True)
        
        # compute the bag feature
        bag_feat = torch.mm(bag_A, h)
        return torch.stack(outputs, dim=0), self.bag_classifier(bag_feat), A_raw.unsqueeze(0)
    