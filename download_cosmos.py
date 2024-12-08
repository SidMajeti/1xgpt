import torch
from cosmos_tokenizer.video_lib import CausalVideoTokenizer

model_name = "Cosmos-Tokenizer-CV4x8x8"
input_tensor = torch.randn(1, 3, 9, 512, 512).to('cuda').to(torch.bfloat16)  # [B, C, T, H, W]
encoder = CausalVideoTokenizer(checkpoint_enc=f'pretrained_ckpts/{model_name}/encoder.jit')
(latent,) = encoder.encode(input_tensor)
torch.testing.assert_close(latent.shape, (1, 16, 3, 64, 64))

# The input tensor can be reconstructed by the decoder as:
decoder = CausalVideoTokenizer(checkpoint_dec=f'pretrained_ckpts/{model_name}/decoder.jit')
reconstructed_tensor = decoder.decode(latent)
torch.testing.assert_close(reconstructed_tensor.shape, input_tensor.shape)