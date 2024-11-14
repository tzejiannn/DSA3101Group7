import os
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from VAE_model import VAE, load_vae_model  # Import the VAE model class and load_vae_model function
from PIL import Image
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Define a default model folder where the models are stored
MODEL_PATH = 'vae_models/'  # Folder where all models are saved

# Route to list available models
@app.route('/available_models', methods=['GET'])
def available_models():
    # List all the .pth files in the vae_models directory
    models = [f.replace('.pth', '') for f in os.listdir(MODEL_PATH) if f.endswith('.pth')]
    return jsonify({"models": models})


# Route to generate an image from a random latent vector
@app.route('/generate', methods=['GET'])
def generate_image():
    # Get the model name from the request arguments
    model_name = request.args.get('model_name', default='vae_64', type=str)  # Default to 'vae_64' if not provided

    try:
        # Load the corresponding VAE model
        vae_model = load_vae_model(model_name, MODEL_PATH)
    except FileNotFoundError:
        return jsonify({"error": "Model not found"}), 404

    # Generate a random latent vector (for demonstration)
    latent_sample = torch.randn(1, 64)  # Adjust the latent dimension if needed

    # Generate the image from the latent vector
    with torch.no_grad():
        generated_image = vae_model.decode(latent_sample).cpu()

    # Convert to a PIL image and save to a buffer
    generated_image = generated_image.view(28, 28)  # For example, using 28x28 images
    generated_image = generated_image.numpy() * 255
    generated_image = Image.fromarray(generated_image.astype(np.uint8))

    # Save the image to a buffer
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Return the image as a response
    return img_byte_arr, 200, {'Content-Type': 'image/png'}


# Route to handle image uploads and return VAE-generated output
@app.route('/generate_from_input', methods=['POST'])
def generate_from_input():
    # Get the model name from the request arguments (or body)
    model_name = request.args.get('model_name', default='vae_64', type=str)

    # Load the model
    try:
        vae_model = load_vae_model(model_name, MODEL_PATH)
    except FileNotFoundError:
        return jsonify({"error": "Model not found"}), 404

    # Get the image from the request (as raw image data)
    image_file = request.files.get('image')

    if image_file:
        # Process the image as before
        img = Image.open(image_file)
        img = img.convert('L')  # Convert to grayscale (adjust if your VAE uses color images)
        img = np.array(img) / 255.0  # Normalize to [0, 1]

        # Convert image to tensor
        img_tensor = torch.tensor(img, dtype=torch.float32).view(1, -1)  # Flatten image to 1D vector

        # Pass through the VAE encoder and decoder
        with torch.no_grad():
            mu, logvar = vae_model.encode(img_tensor)
            z = vae_model.reparameterize(mu, logvar)
            generated_image = vae_model.decode(z).cpu().view(28, 28)

        # Convert generated image to PIL format
        generated_image = (generated_image.numpy() * 255).astype(np.uint8)
        generated_image = Image.fromarray(generated_image)

        # Save the image to a buffer
        img_byte_arr = io.BytesIO()
        generated_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return img_byte_arr, 200, {'Content-Type': 'image/png'}
    else:
        return jsonify({"error": "No image provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)
