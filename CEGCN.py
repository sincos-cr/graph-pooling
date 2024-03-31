import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim, drop_p,adjacency_matrix):
        super(GraphUnet, self).__init__()

     
        self.ks = ks         
        self.bottom_gcn = GCN(dim, dim,  drop_p,adjacency_matrix)        
        self.down_gcns = nn.ModuleList()       
        self.up_gcns = nn.ModuleList()
        self.cross_cnns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
      
        self.l_n = len(ks)
        
        
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim,  drop_p, adjacency_matrix))
            self.up_gcns.append(GCN(dim, dim,  drop_p, adjacency_matrix))
            self.cross_cnns.append(CNN1(dim, dim, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))        
     

    def forward(self, g, h):
        adj_ms = [] 
        indices_list = [] 
        down_outs = [] 
        hs = [] 
        org_h = h 
        
        
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h) 
            adj_ms.append(g) 
            down_outs.append(h) 
            g, h, idx = self.pools[i](g, h) 
            indices_list.append(idx) 
        h = self.bottom_gcn(g, h) 
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1 
            g, idx = adj_ms[up_idx], indices_list[up_idx] 
            
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)             
           
            cross = self.cross_cnns[i](h, down_outs[up_idx])           
            h = self.up_gcns[i](g, cross)           

        h = h.add(org_h) 
        
        return h, hs, g




class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, p,adjacency_matrix):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.BN = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)
        self.adj = adjacency_matrix
        self.I = torch.eye(adjacency_matrix.shape[0], adjacency_matrix.shape[0], requires_grad=False, device=device, dtype=torch.float32)
        self.mask = torch.ceil(adjacency_matrix * 0.00001)
        self.lambda_ = nn.Parameter(torch.zeros(1))

    def A_to_D_inv(self, g):
        D = g.sum(1)
        D_hat = torch.diag(torch.pow(D, -0.5))
        return D_hat
        
    def forward(self, g, h):
        
        h = self.BN(h)
        
        g = g + torch.eye(g.shape[0], g.shape[0], requires_grad=False, device=device, dtype=torch.float32)
        D_hat = self.A_to_D_inv(g)
        g = torch.matmul(D_hat, torch.matmul(g,D_hat))
        h = self.proj(h)
        h = torch.matmul(g, h)
        
        h = self.act(h)
        return h


class CNN1(nn.Module):

    def __init__(self, in_dim, out_dim, p):
        super(CNN1, self).__init__()
        self.BN = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act = nn.LeakyReLU(inplace=True)
        
        self.conv1_1 = nn.Conv1d(in_dim,in_dim,1)

        
    def forward(self, h, pre_h):
        
        h = self.BN(h)
        pre_h = self.BN(pre_h)
        h = h.unsqueeze(0)
        h = h.permute(0,2,1) 
        h1 = self.conv1_1(h) 
        h2 = self.conv1_1(h)
        pre_h = pre_h.unsqueeze(0)
        pre_h = pre_h.permute(0,2,1)
        h3 = self.conv1_1(pre_h)

        h1 = h1.squeeze(0) 
        h2 = h2.squeeze(0) 
        h2 = h2.permute(1,0) 
        h3 = h3.squeeze(0) 
        h3 = h3.permute(1,0) 

        sim = h2.mm(h1) 
        sim = F.softmax(sim)
        sim = sim.mm(h3) 
        sim = F.softmax(sim)
        sim = sim.mul(h3) + h2 + h3
        
        return sim



class GCN11(nn.Module):

    def __init__(self, in_dim, out_dim, p):
        super(GCN11, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.BN = nn.BatchNorm1d(in_dim)
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()
        self.act1 = nn.Sigmoid()
        self.act2 = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.BN(x)
        x = self.proj(x)
        x = self.drop(x)
        
        x = self.act2(x)
        return x






class Pool(nn.Module):

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k 
        self.sigmoid = nn.Sigmoid()
        self.BN = nn.BatchNorm1d(in_dim)
        self.softmax = nn.Softmax()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.sigma1= torch.nn.Parameter(torch.tensor([0.2],requires_grad=True))

    def forward(self, g, h):

        
        h = self.BN(h)
        D = g.sum(1)
        D_hat = torch.diag(torch.pow(D, -1))
        Z1 = torch.abs(h - torch.matmul(D_hat, torch.matmul(g,h))).sum(dim=1)
        Z2 = torch.sum(g,dim=1)       
        pl = self.sigmoid(Z1 + Z2)        
        Z3 = torch.matmul(D_hat, torch.matmul(g,h)) 
        Z3 = self.proj(Z3).squeeze() 
        pg = F.softmax(Z3)    
        pt = self.sigmoid(pl + pg) 
        weights = self.proj(h).squeeze()
        pf = self.sigmoid(weights)       
        scores = self.sigma1*pt + (1-self.sigma1)*pf    

        return top_k_graph(scores, g, h, self.k)     


class Unpool(nn.Module):

    def __init__(self, in_dim, out_dim, p):
        super(Unpool, self).__init__()
        

    def forward(self, g, h, pre_h, idx):
        
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
      
        return g, new_h


def top_k_graph(scores, g, h, k):
    num_nodes = g.shape[0]  
    
    values, idx = torch.topk(scores, max(2, int(k*num_nodes)))
    score1 = scores.expand(h.shape[1],scores.shape[0])
    score1 = score1.t()
    h_att = score1.mul(h) 
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    a_hat = un_g[idx, :] 
    new_h = a_hat.mm(h_att)
    
    
    g_new = a_hat[:, idx] 
    g_new = norm_g(g_new)
   
    return g_new, new_h, idx

def norm_g(g):
    degrees = torch.sum(g, 1)
    g = g / degrees
    return g




class Initializer(object):

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)



class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch//2,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch//2,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU(inplace=True)
        self.Act2 = nn.LeakyReLU(inplace=True)
        self.BN=nn.BatchNorm2d(in_ch)
        
    
    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out


class TFAP(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, model='normal'):
        super(TFAP, self).__init__()
       
        self.class_count = class_count 
        
        self.channel = changel
        self.height = height
        self.width = width
        self.Q = Q
        self.A = A
        self.model=model
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  
        
        self.ks = torch.tensor([0.8, 0.7])
        
        self.act = nn.LeakyReLU(inplace=True)
        
        nodes_count=self.A.shape[0]
        self.I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        layers_count=2
        
        
        
        self.s_gcn = GCN(50, 50, 0.0, self.A) 
        self.g_unet = GraphUnet(self.ks, 50, 50, 50,  0.0, self.A)
        self.l_gcn = GCN(50, 50, 0.0, self.A)

       
        self.conv2 = SSConv(50, 50, kernel_size=5)
        self.conv3 = SSConv(50, 50, kernel_size=5)
        self.conv4 = SSConv(50, 50, kernel_size=5)
        self.conv5 = SSConv(50, 50, kernel_size=5)

        
        
        self.Softmax_linear =nn.Sequential(nn.Linear(50, self.class_count)) #IP:
    
    
    def norm_g(g):
        degrees = torch.sum(g, 1)
        g = g / degrees 
        return g

    def forward(self, x: torch.Tensor,showFlag=False):
        
        (hei, wid, c) = x.shape 
       
        clean_x=x
        clean_x_flatten = clean_x.reshape([hei * wid, -1]) 
        
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  
  
        H = superpixels_flatten 
        
        A_ = norm_g(self.A) 
       
        
        H = self.s_gcn(A_, H)
       
        h, hs, g = self.g_unet(A_, H) 
       
        hs = self.l_gcn(g, h+superpixels_flatten)   
        
        fuse_feature = hs + H+superpixels_flatten

        GCN_result = torch.matmul(self.Q, fuse_feature)  
        
        GCN_result = GCN_result.reshape([hei, wid, -1]) 
        GCN_result = torch.unsqueeze(GCN_result.permute([2, 0, 1]), 0)
        cnn1 = self.conv2(GCN_result)
        res1 = cnn1 + GCN_result
        cnn2 = self.conv3(res1)
        res2 = res1 + cnn2
        cnn3 = self.conv4(res2) 
        out = torch.squeeze(out, 0).permute([1, 2, 0]).reshape([hei * wid, -1])
        Y = self.Softmax_linear(out)
        Y = F.softmax(Y, -1) 
        return Y