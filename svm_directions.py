# %%
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

split_count = 100
confident_count = 400
# %%
def create_image_latent_df(img_dir, lat_dir):
    imgs = os.listdir(img_dir)
    lats = os.listdir(lat_dir)
    
    images = [file for file in imgs]
    latents = [file for file in lats]
    
    images.sort()
    latents.sort()
    return pd.DataFrame({"image_id":images, "latent_id":latents})

df = create_image_latent_df("/kaggle/input/stylegan2-generated/sgan2-generated/images", "/kaggle/input/stylegan2-generated/sgan2-generated/latents")

# %%
columns = ["Bald", "Black_Hair", "Blond_Hair", "Eyeglasses", "Goatee", "Male", "Mustache", "Smiling"]

model = models.resnext50_32x4d(pretrained=True)
for i, (name, children) in enumerate(model.named_children()):
    if(i <= 10):
        print(f"param:{name}, i:{i} freezed")
        children.requires_grad = False
    
    
num_ftrs = model.fc.in_features

model.fc = nn.Linear(num_ftrs, len(columns))

# %%
weight_path = "/kaggle/input/celeba_selected/pytorch/last/1/resnet50_celeba_weights.pth"

# %%
model.load_state_dict(torch.load(weight_path))

# %%
for col in columns:
    df[col] = 0.0

# %%
df_split = np.array_split(df, split_count)
model.to(torch.device("cuda"))
a = []

for df_i in df_split: 
    transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_image_mem(base,img_path):
        image = Image.open(base+img_path).convert("RGB")
        return transform(image)

    image_list = df_i["image_id"]
    images = torch.stack([load_image_mem("/kaggle/input/stylegan2-generated/sgan2-generated/images/", path) for path in image_list])
    images = images.to(torch.device("cuda"))
    with torch.no_grad():
        outputs = model(images)
        outputs = torch.sigmoid(outputs)
        a.append(outputs)
    


# %%
all_predictions = torch.cat(a, dim=0)


all_predictions_np = all_predictions.cpu().numpy()

predictions_df = pd.DataFrame(all_predictions_np, columns=columns)

# %%
df_copy = df.copy()

# %%
df_copy[columns] = predictions_df

# %%
dfs = {}
for col in columns:
    dfs[col] = df_copy[["image_id", "latent_id", col]]

# %%
for col, df in dfs.items():
    dfs[col] = df.sort_values(by=col, ascending=False).reset_index(drop=True)

# %%


# %%
#500k/10k -> 20k/400
highest = {}
lowest = {}

# %%
for col, df in dfs.items():
    highest[col] = df.head(confident_count)
    lowest[col] = df.tail(confident_count)

# %%
print(highest)
print(lowest)

# %%
def load_latents(base_path, latent_ids):
    latents = []
    for latent_id in latent_ids:
        latent_path = os.path.join(base_path, latent_id.strip())
        latent = np.load(latent_path)
        latents.append(latent.flatten())
    return np.array(latents)

latent_base_path = "/kaggle/input/stylegan2-generated/sgan2-generated/latents"

svm_models = {}

for col in columns:
    latent_ids_high_conf = highest[col]['latent_id']
    latent_ids_low_conf = lowest[col]['latent_id']
    
    latents_high_conf = load_latents(latent_base_path, latent_ids_high_conf)
    latents_low_conf = load_latents(latent_base_path, latent_ids_low_conf)
    
    latents = np.vstack((latents_high_conf, latents_low_conf))
    
    labels = np.array([1] * confident_count + [0] * confident_count)
    print(latents.shape)

    scaler = StandardScaler()
    latents_scaled = scaler.fit_transform(latents)
    
    svm = SVC(kernel='linear', C=1.0)
    svm.fit(latents_scaled, labels)
    
    svm_models[col] = svm

# %%
latent_directions = {col: svm.coef_[0] for col, svm in svm_models.items()}


