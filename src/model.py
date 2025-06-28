import torch
from torch import nn
from colorama import Fore 
from einops.layers.torch import Rearrange

class Transformer(nn.Module): 
    def __init__(self, embed_dim, num_heads, mlp_dim, batch_first=True): 
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), 
            nn.GELU(), 
            nn.Linear(mlp_dim, embed_dim) 
        )
        
    def forward(self, x): 
        z1 = self.norm1(x) 
        z1 = self.mha(z1,z1,z1,need_weights=False)
        z1 = z1[0] + x 
        z2 = self.norm2(z1) 
        z2 = self.mlp(z2)  
        return z2 + z1 
    
class Encoder(nn.Module): 
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers): 
        super().__init__() 
        self.blocks = nn.ModuleList([Transformer(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.blocks: 
            x = layer(x)
        return x 

def pos_encoding(seq_length, dim_size): 
    p = torch.zeros((seq_length, dim_size)) 
    for k in range(seq_length):  
        for i in range(int(dim_size/2)): 
            p[k,2*i] = torch.sin(torch.tensor(k/10000 ** (2*i/dim_size)))
            p[k,2*i+1] = torch.cos(torch.tensor(k/10000 ** (2*i/dim_size)))
    return p 

# Build the Model 
class ViT(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.patch = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=8, p2=8)
        self.class_token = nn.Parameter(torch.randn(1,1,1032))
        self.embedding = nn.Linear(192,1032)
        self.register_buffer('positional_embedding', pos_encoding(136,1032)) 
        self.emb_dropout = nn.Dropout(0.25)
        self.norm1 = nn.LayerNorm(192) 
        
        self.norm2 = nn.LayerNorm(1032) 

        self.encoder = Encoder(1032, 12, 2048, 12) 
        self.mlp1 = nn.Linear(1032, 6)
        
    def forward(self,x): 
        batch_size = x.shape[0]
        x = self.patch(x) 
        x = self.norm1(x) 
        x = self.embedding(x)
        x = self.norm2(x) 

        class_tokens = self.class_token.expand(batch_size, -1, -1)
        patch_embedding = torch.cat([class_tokens, x], dim=1)
        final_embedding = patch_embedding + self.positional_embedding 
        final_embedding_dropout = self.emb_dropout(final_embedding) 

        encoded_embedding = self.encoder(final_embedding_dropout) 
        attended_class_token = encoded_embedding[:,0,:]
        z1 = self.mlp1(attended_class_token)

        return z1

if __name__ == '__main__': 
    model = ViT()
    print(model) 
    model.eval()
    print(model(torch.randn(1,3,72,120)))