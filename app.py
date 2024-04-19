import io
import os
from PIL import Image
import chromadb
import torch
import open_clip
from concurrent.futures import ThreadPoolExecutor

# Load configuration from environment variables
CONFIG = {
    "model": os.getenv("MODEL", "ViT-SO400M-14-SigLIP-384"),
    "model_name": os.getenv("MODEL_NAME", "siglip-so400m/14@384"),
    "device": os.getenv("DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"),
}

# Init model
device = torch.device(CONFIG["device"])
pretrained_models = dict(open_clip.list_pretrained())
model_name = CONFIG["model"]
if model_name not in pretrained_models:
    raise ValueError(f"Model {model_name} is not available in pretrained models.")
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, device=device, pretrained=pretrained_models[model_name], precision="fp16")
model.eval()
model.to(device).float()  # Move model to device and convert to half precision
tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')

#chromadb
client = chromadb.PersistentClient(path='./chromadb')
collection = client.get_or_create_collection(name='images')

def file_generator(directory):
    """
    Generates file paths for all files in the specified directory and its subdirectories.
d
    :param directory: The directory path to search for files.
    :return: A generator yielding file paths.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            yield os.path.join(root, file)

def process_image(file_path):
    """
    Processes an image file by extracting metadata and inserting it into the database.

    :param file_path: The path to the image file.
    """
    file = os.path.basename(file_path)
    item = collection.get(ids=[file])
    if item['ids'] !=[]:
        return
    with open(file_path, 'rb') as f:
        file_content = f.read()
    embeddings= get_image_embeddings(file_content)
    collection.add(
        embeddings=embeddings,
        documents=[file],
        ids=[file]
    )

def get_image_embeddings(image_bytes):
    """
    Generate embeddings for an image in bytes.
    """
    # Load and preprocess the image
    image = Image.open(io.BytesIO(image_bytes))
    image = preprocess(image).unsqueeze(0).to(device).float()  # Process image, then move and convert

    # Generate embeddings
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        embeddings = image_features.cpu().numpy().tolist()

    return embeddings

files = list(file_generator('images'))
with ThreadPoolExecutor() as executor:
    futures = []
    for file_path in files:
        if file_path.lower().endswith(('.png','.jpg')):
            future = executor.submit(process_image, file_path)
            futures.append(future)
    for future in futures:
        future.result()
with torch.no_grad():
    text_features = model.encode_text(tokenizer(['a cat']))
    text_features /= text_features.norm(dim=-1, keepdim=True)
results = collection.query(
    query_embeddings=text_features.cpu().numpy().tolist(),
    n_results=10)
print(results)

with torch.no_grad():
    text_features = model.encode_text(tokenizer(['a car']))
    text_features /= text_features.norm(dim=-1, keepdim=True)
results = collection.query(
    query_embeddings=text_features.cpu().numpy().tolist(),
    n_results=10)
print(results)

with torch.no_grad():
    text_features = model.encode_text(tokenizer(['a diagram']))
    text_features /= text_features.norm(dim=-1, keepdim=True)
results = collection.query(
    query_embeddings=text_features.cpu().numpy().tolist(),
    n_results=10)
print(results)