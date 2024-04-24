from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch
import os
import base64
from io import BytesIO

app = Flask(__name__)

# Model path
model_id = r"D:\INTERNSHIP\projects\abhishek\stable-diffusion-v1-5"

# Load model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

@app.route('/generate_image', methods=['POST'])
def generate_image():
    # Get prompt from request data
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({'error': 'Prompt not provided'}), 400

    # Generate image
    image = pipe(prompt).images[0]  

    # Define filename based on prompt
    prompt_filename = "_".join(prompt.split()) + ".png"

    # Define output directory
    output_directory = "output_directory"

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Construct full output path
    output_path = os.path.join(output_directory, prompt_filename)

    # Save image
    image.save(output_path)

    # Convert image to base64
    with open(output_path, "rb") as f:
        image_data = f.read()
        image_base64 = base64.b64encode(image_data).decode()

    return jsonify({'image_base64': image_base64})

if __name__ == '__main__':
    app.run(debug=True)
