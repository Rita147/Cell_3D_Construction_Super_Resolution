

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# In[2]:


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        return self.model(x)


# In[3]:


class HistopathologySRDataset(Dataset):
    def __init__(self, image_dir, transform_hr, transform_lr):
        self.image_dir = image_dir
        self.image_filenames = [f for f in os.listdir(image_dir) if f.endswith('.png') or f.endswith('.jpg')]
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        img = Image.open(img_path).convert('RGB')
        hr = self.transform_hr(img)
        lr = self.transform_lr(img)
        return lr, hr


# In[4]:


transform_hr = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

transform_lr = transforms.Compose([
    transforms.Resize((85, 85)),  # x3 downscale
    transforms.Resize((256, 256)),  # back upsample
    transforms.ToTensor()
])


# In[5]:


def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for lr, hr in tqdm(dataloader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")


# In[6]:


def calculate_psnr(sr, hr):
    mse = F.mse_loss(sr, hr)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr.item()


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = HistopathologySRDataset(image_dir='all tissue img', transform_hr=transform_hr, transform_lr=transform_lr)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = SRCNN().to(device)
criterion = nn.L1Loss()  # replace MSELoss

optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_model(model, dataloader, criterion, optimizer, num_epochs=50, device=device)


# In[8]:


import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_sr_results(model, dataset, device, index=0):
    model.eval()
    with torch.no_grad():
        # Verisetinden bir örnek al
        lr, hr = dataset[index]
        lr = lr.unsqueeze(0).to(device)
        sr = model(lr).cpu().squeeze(0)

        # Tensorları görüntüye çevir
        lr_img = TF.to_pil_image(lr.squeeze(0).cpu())
        hr_img = TF.to_pil_image(hr.cpu())
        sr_img = TF.to_pil_image(torch.clamp(sr, 0, 1))  # clamp önemli!

        # Görüntüleri yan yana göster
        plt.figure(figsize=(12,4))
        plt.subplot(1, 3, 1)
        plt.title("Low-Res (Upsampled)")
        plt.imshow(lr_img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Super-Res (SRCNN Output)")
        plt.imshow(sr_img)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("High-Res (Ground Truth)")
        plt.imshow(hr_img)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# In[ ]:


show_sr_results(model, dataset, device, index=0)


# In[ ]:


import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np

def evaluate_sr_quality(model, dataset, device, index=0):
    """
    Verilen model ve dataset ile belirtilen index'teki görüntü için
    PSNR ve SSIM değerlerini hesaplar ve görüntüleri gösterir.
    """
    model.eval()  # Modeli değerlendirme moduna al
    with torch.no_grad():
        # Dataset'ten LR ve HR görüntüyü al
        lr, hr = dataset[index]
        lr = lr.unsqueeze(0).to(device)     # Model inputu için batch dimension ekle
        hr = hr.unsqueeze(0).to(device)

        # Modelden süper çözünürlük çıktısını al
        sr = model(lr).clamp(0.0, 1.0)      # Çıktıyı 0–1 arasına kırp

        # PSNR hesapla (pytorch → numpy çevirip kullanıyoruz)
        sr_np = sr.squeeze().cpu().permute(1, 2, 0).numpy()  # CxHxW → HxWxC
        hr_np = hr.squeeze().cpu().permute(1, 2, 0).numpy()

        psnr = peak_signal_noise_ratio(hr_np, sr_np, data_range=1.0)

        # SSIM hesapla (renkli olduğu için multichannel=True)
        ssim = structural_similarity(hr_np, sr_np, data_range=1.0, channel_axis=2)

        # Sonuçları yazdır
        print(f" PSNR: {psnr:.2f} dB")
        print(f" SSIM: {ssim:.4f}")

        # Karşılaştırmalı görsel göster
        import matplotlib.pyplot as plt
        import torchvision.transforms.functional as TF

        lr_img = TF.to_pil_image(lr.squeeze(0).cpu())
        hr_img = TF.to_pil_image(hr.squeeze(0).cpu())
        sr_img = TF.to_pil_image(sr.squeeze(0).cpu())

        plt.figure(figsize=(12,4))
        plt.subplot(1, 3, 1)
        plt.title("Low-Res (Upsampled)")
        plt.imshow(lr_img)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title(f"Super-Res\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
        plt.imshow(sr_img)
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("High-Res (Ground Truth)")
        plt.imshow(hr_img)
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# In[ ]:


evaluate_sr_quality(model, dataset, device, index=0)


# In[ ]:


evaluate_sr_quality(model, dataset, device, index=5)
evaluate_sr_quality(model, dataset, device, index=11)


# In[ ]:




