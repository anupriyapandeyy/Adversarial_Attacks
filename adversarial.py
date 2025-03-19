import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchattacks import FGSM
import requests

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TRANSFORMS for ResNet input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# LOAD PRE-TRAINED RESNET MODEL
model = torchvision.models.resnet18(pretrained=True)
model = model.to(device)
model.eval()

# DOWNLOAD IMAGENET CLASS LABELS
#response = requests.get("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt")
# imagenet_labels = dict(enumerate(response.text.splitlines()))


with open('imagenet_classes.txt', 'r') as f:
    imagenet_labels = dict(enumerate(f.read().splitlines()))



# DISPLAY IMAGE FUNCTION
def imshow(img_tensor, title=""):
    npimg = img_tensor.cpu().numpy()
    npimg = np.transpose(npimg, (1, 2, 0))  # Convert to HWC for matplotlib
    plt.imshow(np.clip(npimg, 0, 1))  # Clip to ensure valid range
    plt.title(title)
    plt.axis('off')
    plt.show()

image_path = 'panda.png'  
image = Image.open(image_path).convert('RGB')

# Preprocess and move to device
image_tensor = transform(image).unsqueeze(0).to(device)

# Predict 
# Predict 
outputs = model(image_tensor)
_, predicted = torch.max(outputs.data, 1)
original_prediction = imagenet_labels[predicted.item()]  # Fixed line

print(f"Original Prediction: {original_prediction}")
imshow(image_tensor[0], f"Original: {original_prediction}")



