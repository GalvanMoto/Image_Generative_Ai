from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import os

# Initialize Flask application
app = Flask(__name__)

# Define path to the pre-trained model
model_id1 = r"D:\INTERNSHIP\projects\abhishek\dreamlike-diffusion-1.0"

# Load the model
pipe = StableDiffusionPipeline.from_pretrained(model_id1, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Define route for generating images
@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Parse prompt from request
    prompt = request.json.get('prompt', 'parrot')
    
    # Generate image based on prompt
    image = generate_image_from_prompt(prompt)
    
    # Save generated image
    filename = save_image(image, prompt)
    
    # Return filename of generated image
    return jsonify({'filename': filename})

def generate_image_from_prompt(prompt):
    # Generate an image based on the prompt
    image = pipe(prompt).images[0]
    return image

def save_image(image, prompt):
    # Save the generated image with a unique filename based on the prompt
    filename = f'generated_image_{prompt}.png'
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename

if __name__ == '__main__':
    app.run(debug=True)
