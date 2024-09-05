



import torch
import torch.nn as nn
import numpy as np
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
import warnings
warnings.filterwarnings('ignore')

# 这里可以用两个timm模型进行构建我们的结果
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


class TransformerBlock(nn.Module):
    '''
    Transformer Block combines both the attention module and the feed forward module with layer
    normalization, dropout and residual connections. The sequence of operations is as follows :-
    
    Input -> LayerNorm1 -> Attention -> Residual -> LayerNorm2 -> FeedForward -> Output
      |                                   |  |                                      |
      |-------------Addition--------------|  |---------------Addition---------------|
    '''
    
    def __init__(self, 
                 embed_dim, 
                 heads=8,
                 mlp_ratio=4, # mlp为4倍
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 activation=nn.GELU,
                 norm_layer = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(embed_dim)
        self.attn = Attention(embed_dim, heads=heads, 
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 这里可以选择 drop path， 或者Drop out， 在官方代码中使用了Drop path
        self.drop_path = DropPath(drop_path_ratio)
        # self.drop = nn.Dropout(drop_path_ratio)
        self.norm2 = norm_layer(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop_ratio)
    
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes) # 打乱index
    backward_indexes = np.argsort(forward_indexes) # 得到原来index的位置，方便进行还原
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio 

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape # length, batch, dim
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes) # 随机打乱了数据的patch，这样所有的patch都被打乱了
        patches = patches[:remain_T] #得到未mask的pacth [T*0.25, B, C]

        return patches, forward_indexes, backward_indexes


shuffle = PatchShuffle(0.75)
a = torch.rand(16, 2, 10)
b, forward_indexes, backward_indexes = shuffle(a)
print(b.shape)


class Patchify(torch.nn.Module):
    '''
        把输入进行patch话,为了减少token的数量
        b*5*12*90->(b*12)*5*90 
        conv1D  kernel_size=2 emb_dim=192
        (b*12)*192*45->b*192*12*45
    '''
    def __init__(self,
                 patch_size=2,
                 kernel_size=2,
                 emb_dim=48,
                 in_channels=4,
                 ) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=emb_dim,kernel_size=kernel_size,stride=patch_size,padding=0)

    def forward(self, input):
        B, C, H, W = input.shape # b, 5, 12,90
        input = rearrange(input, 'b c h w -> (b h) c w')
        # print('input.shape',input.shape)
        o_patches = self.conv1(input) # b*12, c, 45
        o_patches = rearrange(o_patches, '(b h) c w  -> b c h w',h=12)

        return o_patches


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 input_height=12,
                 input_weight=90,
                 patch_size=2,
                 kernel_size=2,
                 in_channels=4,
                 emb_dim=48,
                 num_layer=4,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim)) 
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(input_height * input_weight // patch_size, 1, emb_dim))
        # 对patch进行shuffle 和 mask
        self.shuffle = PatchShuffle(mask_ratio)
        
        # # 这里得到一个 (3, dim, patch, patch)
        # self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)
    
        self.patchify=Patchify(patch_size=patch_size,kernel_size=kernel_size,in_channels=in_channels,emb_dim=emb_dim)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        
        # ViT的laynorm
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()
    # 初始化类别编码和向量编码
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        #img b*5*12*90
        patches = self.patchify(img) #patches b*emb_dim*12*45
        # print('patches.shape',patches.shape)
        # os._exit()
        patches = rearrange(patches, 'b c h w -> (h w) b c')  #(12*45)*b*emb_dim
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')  #(1+12*45*(1-mask_ratio))*b*emb_dim

        return features, backward_indexes


class Patch2img(torch.nn.Module):
    '''
        从patches恢复出原图
        rearrange
        (12*45)*b*2->b*12*90
        unsqueeze
        b*12*90->b*1*12*90
    '''
    def __init__(self,
                 height=12,
                 ) -> None:
        super().__init__()
        self.h=height

    def forward(self, input):
        T,B,C = input.shape # (12*45)*b*2
        input = rearrange(input, '(h w) b c -> b h w c',h=self.h )  #b*12*45*2
        input = rearrange(input, 'b h w c -> b h (w c)' )  #b*12*90
        input=input.unsqueeze(1)  #b*1*12*90

        return input
    
patch2img = Patch2img(height=12)
a = torch.rand((12*45), 5, 2)
b= patch2img(a)
print(b.shape)



class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 input_height=12,
                 input_weight=90,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        # self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros(input_height *input_weight // patch_size + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        #(12*45)*b*emb_dim  ->  (12*45)*b*2
        self.head = torch.nn.Linear(emb_dim, patch_size)
        #(12*45)*b*2->b*12*90->b*1*12*90
        self.patch2img = Patch2img(height=input_height)
        # self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]  #features  (1+12*45*(1-mask_ratio))*b*emb_dim
        #因为在patch最开始加了一个cls_token，所以要在backward_indexes之前加上0，其余数字+1
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        #加上被遮盖的patch的编码
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding # 加上了位置编码的信息  #features  (1+12*45)*b*emb_dim

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')   
        features = features[1:] # remove global feature 去掉全局信息，得到图像信息 #features  (12*45)*b*emb_dim

        patches = self.head(features) # 用head得到patchs   (12*45)*b*2
        mask = torch.zeros_like(patches) 
        mask[T:] = 1  # mask其他的像素全部设为 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches) # 得到 重构之后的 img  (12*45)*b*2->b*12*90->b*1*12*90
        mask = self.patch2img(mask)

        return img, mask



class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 input_height=12,
                 input_weight=90,
                 patch_size=2,
                 kernel_size=2,
                 in_channels=4,
                 emb_dim=48,
                 encoder_layer=4,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.1,
                 ) -> None:
        super().__init__()
        
        # print('//////////////////////////////////////////mask_ratio',mask_ratio)
        self.encoder = MAE_Encoder(image_size,input_height,input_weight, patch_size, kernel_size, in_channels, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, input_height,input_weight, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features,  backward_indexes)
        return predicted_img, mask




import random
import torch
import numpy as np
def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
seed = 2022
setup_seed(seed)

