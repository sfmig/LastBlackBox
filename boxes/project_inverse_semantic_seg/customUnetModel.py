import torch
from torch import nn

from diffusers import UNet2DModel




class ClassConditionedUnet(nn.Module):
    '''
    https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb#scrollTo=_7zpDvFxmQeC
    '''
    def __init__(self, masks_size=5): #, num_classes=5, class_emb_size=4):
        super().__init__()

        # The embedding layer will map the class label to a vector of size class_emb_size --(it's used 'separately')
        # self.class_emb = nn.Embedding(num_classes, class_emb_size)

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
        self.model = UNet2DModel(
            sample_size=28,           # 28 #the target image resolution
            in_channels=3 + masks_size, # Additional input channels for class cond.------------------
            out_channels=3,           # the number of output channels (3, for RGB?)
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64), 
            down_block_types=( 
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ), 
            up_block_types=(
                "AttnUpBlock2D", 
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
            ),
        )

    # Our forward method now takes the class labels as an additional argument (to guide the image generation)
    def forward(self, x, t, y_masks): # class_labels):
        # Shape of x:
        # bs, ch, w, h = x.shape
        
        # class conditioning in right shape to add as additional input channels ----> check!
        # class_cond = self.class_emb(class_labels) # Map to embedding dinemsion
        # class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
        # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

        # Net input is now x and class cond concatenated together along dimension 1
        net_input = torch.cat((x, y_masks), 1) #torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

        # Feed this to the unet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 28, 28)