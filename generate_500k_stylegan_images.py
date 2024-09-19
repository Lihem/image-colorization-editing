import torch
from models.stylegan2.model import Generator  # Assuming this is the path to the Generator class
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import ToPILImage
import PIL
import matplotlib.pyplot as plt
from utils import common, train_utils
import math

            
# Load the model
device = "cuda" 
model_path = "pretrained_models/stylegan2-ffhq-config-f.pt" 

model = Generator(1024, 512, 8).to(device)

state_dict = torch.load(model_path)
face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
model.load_state_dict(state_dict['g_ema'], strict=False)

model.eval()

# Function to generate and save image and latent code
def generate_and_save(latent_z, filename):
    # Generate image from latent code (assuming get_all_latents is the function)
    img, w_plus = model(styles=[latent_z], input_is_latent=False, randomize_noise=False, return_latents=True)   # Adapt this line according to the actual function

    img = face_pool(img)

    img = common.tensor2im(img[0])

    resize_amount = (256, 256)
    img.resize(resize_amount)

    img.save(filename)

    w_plus = w_plus[0].detach().cpu().numpy()

    # Save the latent code
    np.save(f"/home/melih.cosgun/GENERATIVE/test_latents/{str(i).zfill(6)}.npy" , w_plus)


# Generate 500,000 images (adapt latent generation based on the model)
num_images = 1

print("Generating 500,000 images with latent codes...")
for i in range(num_images):
    # Generate random latent code (adapt according to how the model expects it)
    print(i)
    latent_z = np.random.randn(1, 512).astype("float32")
    
    # Generate and save image and latent code
    generate_and_save(torch.from_numpy(latent_z).to("cuda"), f"/home/melih.cosgun/GENERATIVE/test_images/{str(i).zfill(6)}.png")

print("Generated 500,000 images with latent codes!")