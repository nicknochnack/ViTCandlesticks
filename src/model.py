# Used for math functions in sinusoidal embedding
import torch
# Main nn layers, MHA and linear
from torch import nn
# Debugging in 4k COLOuR (autralian spelling) 
from colorama import Fore
# Used to create image patches.
from einops.layers.torch import Rearrange
# Pretty print model metrics and layer structure
from torchinfo import summary
# Bring in matplotlib to visualise positional embedding 
from matplotlib import pyplot as plt

# Build MLP Head for Transformer Encoder
class MLPBlock(nn.Sequential):
    # Initialise block
    def __init__(self, embed_dim, mlp_dim):
        # Return MLP Block
        super(MLPBlock, self).__init__(
            # Which takes in a matrix of embed dim, scales up to MLP dim
            nn.Linear(embed_dim, mlp_dim),
            # Applies GELU non-linearity
            nn.GELU(),
            # Then a bit of dropout for helping with variance
            nn.Dropout(0.25),
            # Scales back to embed_dim
            nn.Linear(mlp_dim, embed_dim),
            # And some more dropout
            nn.Dropout(0.25),
        )


# Build a single Encoder Block
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, batch_first=True):
        super().__init__()
        # Apply layer norm
        self.ln_1 = nn.LayerNorm(embed_dim)
        # Then create a MHA attenion layer - effectively self attention 
        self.self_attention = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first, dropout=0.25
        )
        # Create dropout layer
        self.dropout = nn.Dropout(0.25)
        # Another ln layer 
        self.ln_2 = nn.LayerNorm(embed_dim)
        # And use the MLP block 
        self.mlp = MLPBlock(embed_dim, mlp_dim)

    def forward(self, x):
        # Apply layer norm
        z1 = self.ln_1(x)
        # Pass through input as qkv to apply self attention and create skip connection by adding x
        z1 = self.self_attention(z1, z1, z1, need_weights=False)[0] + x
        # Apply dropout
        z1 = self.dropout(z1)
        # Apply layer norm
        z2 = self.ln_2(z1)
        # Pass attended outputs to the MLP block
        z2 = self.mlp(z2) + z1
        # Return z2
        return z2

# Create stacked encoder
class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, num_layers):
        super().__init__()
        # Stack a bunch of the encoder layers together. 
        self.layers = nn.Sequential()
        # Note we're looping through num_layers so more layers = more depth
        for x in range(num_layers):
            self.layers.add_module(
                f"encoder_layer_{x}", EncoderBlock(embed_dim, num_heads, mlp_dim)
            )

    def forward(self, x):
        # Send input to Encoder layers 
        return self.layers(x)

# Create positional encoding 
def pos_encoding(seq_length, dim_size):
    p = torch.zeros((seq_length, dim_size))
    for k in range(seq_length):
        for i in range(int(dim_size / 2)):
            p[k, 2 * i] = torch.sin(torch.tensor(k / (10000 ** (2 * i / dim_size))))
            p[k, 2 * i + 1] = torch.cos(torch.tensor(k / (10000 ** (2 * i / dim_size))))
    return p


# Create a ViT
class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        # Create image patches
        self.patch = Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=8, p2=8)
        # Create class token which will be extracted for classification
        self.class_token = nn.Parameter(torch.randn(1, 1, 1024))
        # Create embedding layer - 192 = 8x8x3, 1024 is configurable 
        self.embedding = nn.Linear(192, 1024)
        # Create positional embedding that gets added to each input embedding 
        self.register_buffer("positional_embedding", pos_encoding(118, 1024))
        # Dropout 
        self.emb_dropout = nn.Dropout(0.25)
        # Create layer norms, one after patch and one after embedding
        self.norm1 = nn.LayerNorm(192)
        self.norm2 = nn.LayerNorm(1024)
        # Create Encoder stack
        self.encoder = Encoder(1024, 8, 2056, 3)
        # Create classification head 
        self.mlp1 = nn.Linear(1024, 5)

    def forward(self, x):
        # Create image patches 
        x = self.patch(x)
        x = self.norm1(x)
        # Pass patches to embedding layer
        x = self.embedding(x)
        x = self.norm2(x)
        # print(Fore.LIGHTMAGENTA_EX + str(x.shape) + Fore.RESET)
        # Create class tokens and append them to the start of the sequence 
        batch_size = x.shape[0]
        class_tokens = self.class_token.expand(batch_size, -1, -1)
        patch_embedding = torch.cat([class_tokens, x], dim=1)
        # Apply positional embedding and dropout 
        final_embedding = patch_embedding + self.positional_embedding
        final_embedding_dropout = self.emb_dropout(final_embedding)
        # print(Fore.LIGHTCYAN_EX + str(x.shape) + Fore.RESET)
        # Padd the embedding + positional embedding to the encoder stack
        encoded_embedding = self.encoder(final_embedding_dropout)
        # print(Fore.LIGHTGREEN_EX + str(encoded_embedding.shape) + Fore.RESET)
        # Extract the classification token
        attended_class_token = encoded_embedding[:, 0, :]
        # print(Fore.LIGHTRED_EX + str(attended_class_token.shape) + Fore.RESET)
        # Pass it to the MLP head - this will return logits (which need to have an softmax function applied for probabilities) 
        z1 = self.mlp1(attended_class_token)

        return z1

# Test it out
if __name__ == "__main__":

    stack_components = ['encoder', 'encoder_block', 'mlp_block', 'pos_embedding', 'vit']
    component = stack_components[0]

    if component == 'encoder': 
        encoder = Encoder(1024, 8, 2056, 3)    
        summary(encoder, (1,118,1024))
    elif component == 'encoder_block': 
        encoder_block = EncoderBlock(1024, 8, 2056)    
        summary(encoder_block, (1,118,1024))
    elif component == 'mlp_block': 
        mlp_head = MLPBlock(768, 1024)
        summary(mlp_head, (1, 118, 768))
    elif component == 'pos_embedding': 
        pos = pos_encoding(118, 1024)
        cax = plt.matshow(pos) 
        plt.gcf().colorbar(cax)
        plt.show()
        # plt.savefig('posembedding') 
    elif component =='vit': 
        model = ViT()
        summary(model, (1, 3, 104, 72))
    else:
        pass 

    # model = ViT()
    # example_inputs = torch.randn(4, 3, 104, 72)
    # onnx_program = torch.onnx.export(model, example_inputs, dynamo=True)
    # onnx_program.save("archive/ViTCandlestick.onnx")