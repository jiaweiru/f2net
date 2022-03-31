import torch
import torchaudio

a = torch.rand(3, 3).cuda()
print(a)
b = a.to('cpu')
b = b + 1
print(a)
# a = a * 1.
# length = a.shape[0]
# a1 = a[:-1] if length % 2 == 1 else a
# downsample = torchaudio.transforms.Resample(16000, 8000)
# upsample = torchaudio.transforms.Resample(8000, 16000)
# b = a1[::2]
# b_up = upsample(b)
# print(a.shape, a1.shape, b_up.shape)
