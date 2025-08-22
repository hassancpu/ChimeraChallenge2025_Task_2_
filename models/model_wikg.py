import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GlobalAttention
from utils.utils import initialize_weights

class WiKG(nn.Module):
    def __init__(self, topk=6, n_classes=2, feat_type='uni'):
        super().__init__()
        if feat_type == 'uni':
            self.size = [1024, 512, 256]
        elif feat_type == 'gigapath':
            self.size = [1536, 768, 384]
        self._fc1 = nn.Sequential(nn.Linear(self.size[0], self.size[1]),  nn.ReLU())
        self._fc1.append(nn.Dropout(0.25))

        self.W_head = nn.Linear(self.size[1], self.size[1])
        self.W_tail = nn.Linear(self.size[1], self.size[1])

        self.scale = self.size[1] ** -0.5
        self.topk = topk

        self.gate_U = nn.Linear(self.size[1], self.size[1] // 2)
        self.gate_V = nn.Linear(self.size[1], self.size[1] // 2)
        self.gate_W = nn.Linear(self.size[1] // 2, self.size[1])

        self.linear1 = nn.Linear(self.size[1], self.size[1])
        self.linear2 = nn.Linear(self.size[1], self.size[1])

        
        self.activation = nn.LeakyReLU()
        self.message_dropout = nn.Dropout(0.3)

        self.norm = nn.LayerNorm(self.size[1])
        self.fc = nn.Linear(self.size[1], n_classes)

        att_net=nn.Sequential(nn.Linear(self.size[1], self.size[1] // 2), nn.LeakyReLU(), nn.Linear(self.size[1]//2, 1))     
        self.readout = GlobalAttention(att_net)
        initialize_weights(self)

    def relocate(self):
        """
        Relocates the model to GPU if available, else to CPU.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._fc1= self._fc1.to(device)
        self.W_head = self.W_head.to(device)
        self.W_tail = self.W_tail.to(device)
        self.gate_U = self.gate_U.to(device)
        self.gate_V = self.gate_V.to(device)
        self.gate_W = self.gate_W.to(device)
        self.linear1 = self.linear1.to(device)
        self.linear2 = self.linear2.to(device)
        self.activation = self.activation.to(device)
        self.message_dropout = self.message_dropout.to(device)
        self.norm = self.norm.to(device)
        self.fc = self.fc.to(device)
        self.readout = self.readout.to(device)

    def forward(self, x, coords= None):
        x= torch.unsqueeze(x, 0)
        x = self._fc1(x)    # [B,N,C]

        # B, N, C = x.shape
        x = (x + x.mean(dim=1, keepdim=True)) * 0.5  

        e_h = self.W_head(x)
        e_t = self.W_tail(x)

        # construct neighbour
        attn_logit = (e_h * self.scale) @ e_t.transpose(-2, -1)  # 1
        topk_weight, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1)

        # add an extra dimension to the index tensor, making it available for advanced indexing, aligned with the dimensions of e_t
        topk_index = topk_index.to(torch.long)

        # expand topk_index dimensions to match e_t
        topk_index_expanded = topk_index.expand(e_t.size(0), -1, -1)  # shape: [1, 10000, 4]

        # create a RANGE tensor to help indexing
        batch_indices = torch.arange(e_t.size(0)).view(-1, 1, 1).to(topk_index.device)  # shape: [1, 1, 1]

        Nb_h = e_t[batch_indices, topk_index_expanded, :]  # shape: [1, 10000, 4, 512]

        # use SoftMax to obtain probability
        topk_prob = F.softmax(topk_weight, dim=2)
        eh_r = torch.mul(topk_prob.unsqueeze(-1), Nb_h) + torch.matmul((1 - topk_prob).unsqueeze(-1), e_h.unsqueeze(2))  # 1 pixel wise   2 matmul

        # gated knowledge attention
        e_h_expand = e_h.unsqueeze(2).expand(-1, -1, self.topk, -1)
        gate = torch.tanh(e_h_expand + eh_r)
        ka_weight = torch.einsum('ijkl,ijkm->ijk', Nb_h, gate)

        ka_prob = F.softmax(ka_weight, dim=2).unsqueeze(dim=2)
        e_Nh = torch.matmul(ka_prob, Nb_h).squeeze(dim=2)

        sum_embedding = self.activation(self.linear1(e_h + e_Nh))
        bi_embedding = self.activation(self.linear2(e_h * e_Nh))
        embedding = sum_embedding + bi_embedding
        
        h = self.message_dropout(embedding)

        h = self.readout(h.squeeze(0), batch=None)
        h = self.norm(h)
        h = self.fc(h)
        Y_hat = torch.topk(h, 1, dim=1)[1]
        Y_prob = F.softmax(h, dim=1) 

        results_dict = {} 
        return h, Y_prob, Y_hat, results_dict, results_dict
