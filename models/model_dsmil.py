import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import initialize_weights

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        fc = [nn.Linear(in_size, out_size), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.fc = nn.Sequential(*fc)

    def forward(self, feats):
        x = self.fc(feats)
        return x

class IClassifier(nn.Module):
    def __init__(self, feature_extractor, feature_size, output_class):
        super(IClassifier, self).__init__()
        self.feature_extractor = feature_extractor      
        self.fc = nn.Linear(feature_size, output_class)
        
    def forward(self, x):
        feats = self.feature_extractor(x) # N x K
        c = self.fc(feats.view(feats.shape[0], -1)) # N x C
        return feats.view(feats.shape[0], -1), c

class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0): # K, L, N
        super(BClassifier, self).__init__()
        self.q = nn.Linear(input_size, 128)
        self.v = nn.Sequential(
            nn.Dropout(dropout_v),
            nn.Linear(input_size, input_size)
        )    
        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)  
        
    def forward(self, feats, c): # N x K, N x C
        device = feats.device
        V = self.v(feats) # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1) # N x Q, unsorted
        
        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True) # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :]) # select critical instances, m_feats in shape C x K 
        q_max = self.q(m_feats) # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1)) # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax( A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0) # normalize attention scores, A in shape N x C, 
        B = torch.mm(A.transpose(0, 1), V) # compute bag representation, B in shape C x V
                
        B = B.view(1, B.shape[0], B.shape[1]) # 1 x C x V
        C = self.fcc(B) # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B 
    
class DSMIL(nn.Module):
    def __init__(self, n_classes=2, feat_type= 'imagenet'):
        super(DSMIL, self).__init__()
        if feat_type == 'imagenet':
            self.size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            self.size = [1536, 768, 384]

        self.fc = FCLayer(in_size=self.size[0], out_size=self.size[1])
        self.i_classifier = IClassifier(self.fc ,feature_size= self.size[1], output_class= n_classes)
        self.b_classifier = BClassifier(input_size= self.size[1], output_class= n_classes, dropout_v=0.0)
        initialize_weights(self)
        
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.i_classifier =  self.i_classifier.to(device)
        self.b_classifier = self.b_classifier.to(device)
        self.fc = self.fc.to(device)
        
    def forward(self, x, coords= None):
        feats, classes = self.i_classifier(x)
        prediction_bag, _, _ = self.b_classifier(feats, classes)
        
        max_prediction, _ = torch.max(classes, 0)
        logits = 0.5 * (prediction_bag + max_prediction)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        return logits, Y_prob, Y_hat, 0, 0 

