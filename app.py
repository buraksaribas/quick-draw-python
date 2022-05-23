#virtualenv 
#.\Scripts\activate
#streamlit run app.py

import streamlit as st
from streamlit_drawable_canvas import st_canvas

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import json
# https://pypi.org/project/streamlit-drawable-canvas/

# Specify canvas parameters in application
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
stroke_width = st.sidebar.slider(
    label='Stroke width:',
    min_value=1, 
    max_value=15, 
    value=3
)
drawing_mode = st.sidebar.selectbox(
    label='Drawing tool:', 
    options=('freedraw', 'transform')
)
realtime_update = st.sidebar.checkbox(
    label='Update in realtime', 
    value=True
)


# Create a canvas component
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    update_streamlit=realtime_update,
    height=400,
    width=400,
    drawing_mode=drawing_mode,
    key='canvas',
)



@st.cache
def load_model():
    model = torch.load(
        f='models/model.pt',
        map_location=torch.device('cpu')
    )

    return model

model = load_model()

transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.9720, 0.9720, 0.9720), 
                             (0.1559, 0.1559, 0.1559)) # Normalize with the mean and std of the whole dataset
])

# Dictionary to map id to name of the class
with open('categories/id_to_class.json') as file:
    id_to_class = json.load(file)

if canvas_result.image_data is not None:
    image = canvas_result.image_data
    
    # Convert RGBA image to RGB
    image_rgb = Image.fromarray(np.uint8(image)).convert(mode='P')
    image_rgb = np.array(image_rgb)[:, :, np.newaxis]
    image_rgb = np.repeat(image_rgb, repeats=3, axis=2)
    # Use the same transformation used in training and add batch dimension
    image_rgb = torch.unsqueeze(transform(image_rgb), dim=0)

    # Compute logits
    y_lgts = model(image_rgb)
    # Compute scores
    y_prob = F.softmax(y_lgts, dim=1)
    # Compute the top 5 predictions
    top_5 = torch.topk(y_prob, k=5)
    preds = top_5.indices.numpy().flatten()
    probs = top_5.values.detach().numpy().flatten()
    
    labels = [id_to_class[str(i)] for i in preds]
    predictions = dict(zip(labels, probs))
    
    st.write('**Top 5 predictions:**')
    st.write(predictions)
    #print(list(predictions.keys())[0])
    import requests
    r = requests.post(
        "https://api.deepai.org/api/text2img",
        data={
            'text': list(predictions.keys())[0],
        },
        headers={'api-key': '8af203ea-8407-4e1c-9ed7-a528f85b9da0'}
    )
    
    img = r.json()['output_url']
    st.image(img)

