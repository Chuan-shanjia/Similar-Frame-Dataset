import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import pandas as pd
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model = model.to(device)
model.eval()

def image_contains_person(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    with torch.no_grad():
        prediction = model(image)
    
    for element in prediction[0]['labels']:
        if element.item() == 1:
            return True
    return False

if __name__ == "__main__":
    csv_path = ''  # Replace with your annotation-file's path for query-target candidate images
    df = pd.read_csv(csv_path)
    df['contains_person'] = df['path'].apply(image_contains_person)
    output_csv_path = csv_path
    df.to_csv(output_csv_path, index=False)