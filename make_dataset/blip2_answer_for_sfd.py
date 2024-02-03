import torch
from PIL import Image
import pandas as pd
import time
from transformers import AutoProcessor, Blip2ForConditionalGeneration


processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

def get_caption_from_img(prompt, img_path):
    image = Image.open(img_path)
    inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=1)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model.to(device)

    csv_path = '' # Replace with your annotation-file's path for query-target candidate images
    df = pd.read_csv(csv_path)
    img_path_list = df['path'].tolist()

    for index, row in df.iterrows():
        action_class = row['class']
        img_path = row['path']

        prompt = f"Question: Is the person {action_class}? Answer:"
        
        try:
            generated_text = get_caption_from_img(prompt, img_path)
        except Exception as e:
            print(f"An error occurred: {e}")
            generated_text = 'Error'

        df.at[index, 'blip2_result'] = generated_text
        if index%10==0:
            print(action_class)
            print(generated_text)

    df.to_csv(csv_path, index=False)
