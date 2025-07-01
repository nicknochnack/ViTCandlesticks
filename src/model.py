import torch
from torch import nn
from colorama import Fore 
from einops.layers.torch import Rearrange
import torchvision

class MLPBlock(nn.Sequential): 
    def __init__(self, embed_dim, mlp_dim): 
        super(MLPBlock, self).__init__(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(), 
            nn.Linear(mlp_dim, embed_dim), 
            nn.Dropout()
        )

class EncoderBlock(nn.Module): 
    def __init__(self, embed_dim, num_heads, mlp_dim, batch_first=True): 
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first, dropout=0.5)
        self.dropout = nn.Dropout(0.0)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = MLPBlock(embed_dim, mlp_dim)
    
    def forward(self, x): 

        z1 = self.ln_1(x)
        # Add in pre ln skip
        z1 = self.self_attention(z1,z1,z1,need_weights=False) + x 
        z1 = self.dropout(z1[0])
        
        z2 = self.ln_2(z1)
        # Add in pre ln skip
        z2 = self.mlp(z2) + z1
        return x 

class Encoder(nn.Module): 
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers): 
        super().__init__()
        self.layers = nn.Sequential() 
        for x in range(num_layers): 
            self.layers.add_module(f"encoder_layer_{x}", EncoderBlock(embed_dim, num_heads, mlp_dim))
    def forward(self, x): 
        return self.layers(x) 

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
        self.patch = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=16, p2=16)
        self.class_token = nn.Parameter(torch.randn(1,1,768))
        self.embedding = nn.Linear(768,768)
        self.register_buffer('positional_embedding', pos_encoding(197,768)) 
        self.emb_dropout = nn.Dropout(0.5)
        self.norm1 = nn.LayerNorm(768) 
        self.norm2 = nn.LayerNorm(768) 

        self.encoder = Encoder(768, 12, 3072, 12) 
        self.mlp1 = nn.Linear(768, 6)
        
    def forward(self,x): 
        batch_size = x.shape[0]
        x = self.patch(x) 
        x = self.norm1(x) 
        x = self.embedding(x)
        x = self.norm2(x) 
        # print(Fore.LIGHTMAGENTA_EX + str(x.shape) + Fore.RESET) 
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        patch_embedding = torch.cat([class_tokens, x], dim=1)
        final_embedding = patch_embedding + self.positional_embedding 
        final_embedding_dropout = self.emb_dropout(final_embedding) 
        # print(Fore.LIGHTCYAN_EX + str(x.shape) + Fore.RESET) 
        encoded_embedding = self.encoder(final_embedding_dropout) 
        # print(Fore.LIGHTGREEN_EX + str(encoded_embedding.shape) + Fore.RESET) 
        attended_class_token = encoded_embedding[:,0,:]
        # print(Fore.LIGHTRED_EX + str(attended_class_token.shape) + Fore.RESET) 
        z1 = self.mlp1(attended_class_token)

        return z1

if __name__ == '__main__': 
    model = ViT()

    pretrained = torchvision.models.vit_b_16() 
    print(model.encoder.layers.load_state_dict(pretrained.encoder.layers.state_dict()))
    print(model) 
    # model.eval()
    # print(model(torch.randn(1,3,72,120)))