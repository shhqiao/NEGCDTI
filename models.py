import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def _init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
           m.bias.data.fill_(0.0)

class GraphConvolution(nn.Module):
    def __init__(self,in_feature,out_feature,bias=True):
        super().__init__()
        self.in_feature = in_feature 
        self.out_feature = out_feature 
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)
            
    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output + self

class Decoder(nn.Module):
    def __init__(self,train_W):
        super().__init__()
        self.train_W = nn.Parameter(train_W)
        
    def forward(self,H,drug_num,target_num):
        HR = H[0:drug_num]
        HD = H[drug_num:(drug_num+target_num)]
        supp1 = torch.mm(HR,self.train_W)
        decoder = torch.mm(supp1,HD.transpose(0,1))    
        return decoder 

class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features,
                 out_features,
                 act_layer=nn.GELU,
                 drop=0.):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 norm_layer=nn.LayerNorm,
                 drop_ratio=0.
                 ):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, drop_ratio=drop_ratio)

        self.drop_path = DropPath(drop_ratio) if drop_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            out_features=dim,
            drop=drop_ratio
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Encoder(nn.Module):
    def __init__(self,
                 depth_e1=4,
                 depth_e2=4,
                 in_dim = 256,
                 embed_dim=256,
                 drop_ratio=0.,
                 ):
        super(Encoder, self).__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.fc_xt = nn.Linear(256 * 256, in_dim)

        # protein
        self.embeddings_e1 = nn.Embedding(23, embed_dim)
        self.norm_e1 = norm_layer(embed_dim)
        self.pos_drop_e1 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e1 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_e1 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e1)]
        self.encoder_e1 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e1[i],
                  )
            for i in range(depth_e1)
        ])

        # smile
        self.smile2feature = nn.Linear(512, 256)
        self.embeddings_e2 = nn.Embedding(51, embed_dim)
        self.norm_e2 = norm_layer(embed_dim)
        self.pos_drop_e2 = nn.Dropout(p=drop_ratio)
        self.pos_embed_e2 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        dpr_e2 = [x.item() for x in torch.linspace(0, drop_ratio, depth_e2)]
        self.encoder_e2 = nn.Sequential(*[
            Block(dim=embed_dim,
                  mlp_ratio=4,
                  drop_ratio=dpr_e2[i],
                  )
            for i in range(depth_e2)
        ])

    def forward_features_e2(self, x):
        x = self.embeddings_e2(x)
        x = x.permute(0, 2, 1)
        x = self.smile2feature(x)
        x = x.permute(0, 2, 1)
        x = self.pos_drop_e2(x + self.pos_embed_e2)
        x = self.encoder_e2(x)
        x = self.norm_e2(x)
        x = x.contiguous().view(-1, 256 * 256)
        x = self.fc_xt(x)
        return x

    def forward_features_e1(self, x):
        x = self.embeddings_e1(x)
        x = self.pos_drop_e1(x + self.pos_embed_e1)
        x = self.encoder_e1(x)
        x = self.norm_e1(x)
        x = x.contiguous().view(-1, 256 * 256)
        x = self.fc_xt(x)
        return x

    def forward(self, smiles_feature, pro_feature):
        drug_feature = self.forward_features_e2(smiles_feature)
        target_feature = self.forward_features_e1(pro_feature)
        H = torch.cat([drug_feature, target_feature], 0)
        return H

class NEGCDTI(nn.Module):
    def __init__(self,depth_e1,depth_e2,embed_dim,in_dim,hgcn_dim,train_W,dropout,drop_ratio):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim,hgcn_dim)
        self.gc2 = GraphConvolution(hgcn_dim,hgcn_dim)
        self.gc3 = GraphConvolution(hgcn_dim,hgcn_dim)
        self.decoder = Decoder(train_W)     
        self.dropout = dropout
        self.encoder = Encoder(depth_e1,depth_e2,in_dim,embed_dim,drop_ratio)
        self.relu = nn.ReLU()
    
    def forward(self,smiles_feature,pro_feature,G,drug_num,target_num):
        H = self.encoder(smiles_feature, pro_feature)
        H = self.relu(H)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc1(H,G)
        H = self.relu(H)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc2(H,G)
        H = self.relu(H)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc3(H,G)
        H = self.relu(H)
        decoder = self.decoder(H,drug_num,target_num)        
        
        return decoder, H
