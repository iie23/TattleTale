from typing import List
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class ClipEmbedder:
    def __init__(self):
        self.model_name = "openai/clip-vit-base-patch32"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def embed_images(self, images: List[Image.Image]):
        with torch.no_grad():
            inputs = self.processor(text=None, images=images, return_tensors="pt", padding=True)
            image_embeds = self.model.get_image_features(**inputs.to(self.device)).to("cpu")
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds

    def embed_text(self, text: str):
        """
        Embed the query in a 512 dim vector.

        :param text: The search query - A single text string
        :return: The feature vector
        """
        with torch.no_grad():
            inputs = self.processor(text=text, images=None, return_tensors="pt", padding=True)
            text_embeds = self.model.get_text_features(**inputs.to(self.device)).to("cpu")
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds

clip = ClipEmbedder()

import cv2

cam = cv2.VideoCapture(0)

import json
import sys
import requests
import base64

def notify(target):
    # Webhooks URL
    url = "https://hooks.slack.com/services/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"
     
    # Message you wanna send
    message = (target)
     
    # Title
    title = (f"Update")
     
    # All slack data
    slack_data = {
        "username": "Testing",
        "attachments": [
            {
                "color": "#FF0000",
                "fields": [
                    {
                        "title": title,
                        "value": message,
                        "short": "false",
 
                    }
                ]
            }
        ]
    }
     
    # Size of the slack data
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json",
               'Content-Length': byte_length}
     
    # Posting requests after dumping the slack data
    response = requests.post(url, data=json.dumps(slack_data), headers=headers)

import openai

openai.api_key = "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"

prompt = input("Security Prompt: ")

completion = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [
        {"role": "system", "content": "You are WatchDogGPT. Your task is to take in something that the user wants their security system to look out for, and output a numbered list of positive and negative image descriptors for the camera to tell whether or not that thing has occured. For example, if the user says 'Let me know if my baby wakes up', then your positive image descriptors could be 'awake baby' and 'crying baby', while your negative image desciptors could be 'sleeping baby' and 'resting baby'."},
        {"role": "user", "content": prompt}
    ]
)
response = completion.choices[0].message.content

pos = []
neg = []
i = 1
L = neg
for line in response.splitlines():
    if line.startswith(str(i) + ". ") or line.startswith("1. "):
        if line.startswith("1. "): i = 1
        if i == 1:
            if L is pos: L = neg
            else: L = pos
        L.append(line[3:])
        i += 1

targets = pos + neg

embeds = {}
for target in targets:
    embeds[target] = clip.embed_text(target)[0]

##

ROBUSTNESS = 5

prev_winner = None
num = 0
while True:
    _, img = cam.read()
    #cv2.imshow('Image', img)
    obs = clip.embed_images(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))[0]
    dots = [torch.dot(obs, embeds[target]) for target in targets]
    winner = targets[dots.index(max(dots))]
    if prev_winner is None or prev_winner != winner:
        num = 1
    else:
        num += 1
    if num == ROBUSTNESS:
        notify(winner)
    prev_winner = winner
    #print(winner)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()