# Standard Library Imports
import re
import os
import time

# Progress Bar
from tqdm import tqdm

# Third-Party Libraries
from bs4 import BeautifulSoup
from datasets import Dataset
from markdown import markdown

# Pytorch / HF Embedding Model Imports
import torch
from transformers import AutoModel, AutoTokenizer

# Function to initialize the model and tokenizer (either from local storage or by downloading)
def instantiate_model_and_tokenizer(model_name='', device='cpu', local_model_dir='./local_model'):
    """Initialize and return the tokenizer and model."""
    
    print(local_model_dir)

    # Check if the model and tokenizer exist in local storage
    if os.path.exists(local_model_dir):
        print(f"Loading {model_name} from local storage...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
        model = AutoModel.from_pretrained(local_model_dir)
    else:
        print(f"{model_name} not found in local storage. Downloading...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, trust_remote_code=True)
        
        # Save the tokenizer and model to local storage
        os.makedirs(local_model_dir, exist_ok=True)
        os.makedirs(local_model_dir, exist_ok=True)
        
        tokenizer.save_pretrained(local_model_dir)
        model.save_pretrained(local_model_dir)
    
    # Move model to input device
    model.to(device)
    print(f"{model_name} embedding model loaded to {device}\n")
    
    return tokenizer, model

def markdown_to_text(markdown_string=''):

    """ Converts a markdown string to plaintext so that embeddings are based on clean text """

    # Remove markdown image embeddings prior to conversion for embedding generation:
    # Example: would remove '![](/content/images/1701.00160v1.pdf-23-0.jpg)'
    image_embedding_pattern = r"!\[\].*?\)"
    markdown_string = re.sub(pattern=image_embedding_pattern, repl="", string=markdown_string)

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(text=markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ''.join(soup.find_all(string=True))

    # reduce multiple newlines to one
    text = re.sub(r'\n+', '\n', text).strip()

    return text

def get_dataset_embeddings(dataset: Dataset, batch_size: int, tokenizer, model, device: str, text_column: str = "markdown_text", use_plaintext: bool = True):
    all_embeddings = []
    all_token_counts = []

    # Start tracking time for the entire embedding process
    start_time = time.time()

    # Initialize the tqdm progress bar
    # Here, we calculate the total number of batches (len(dataset) // batch_size)
    num_batches = len(dataset) // batch_size + (1 if len(dataset) % batch_size != 0 else 0)
    progress_bar = tqdm(range(0, len(dataset), batch_size), total=num_batches, desc="Processing batches", unit="batch")

    for start_idx in progress_bar:
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        batch_texts = batch[text_column]

        if use_plaintext:
            batch_texts = [markdown_to_text(text) for text in batch[text_column]]
        
        # Debugging print statement
        # print(f"Processing batch indicies: {start_idx} to {end_idx - 1}")

        # Tokenize and move to device
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        token_counts = tokens["attention_mask"].sum(dim=1).tolist() # Attention mask is 0 for padding, 1 for actual tokens

        # Get embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of vectors and store
        all_embeddings.extend(embeddings.cpu().numpy())
        all_token_counts.extend(token_counts)

    # Add embeddings as new column to the dataset
    dataset = dataset.add_column("embedding", all_embeddings)

    # Modify chunk_metadata in-place to add token_count
    def add_token_count(example, idx):
        example["chunk_metadata"]["text_token_count"] = all_token_counts[idx]
        return example

    # Use map with index to update each example's chunk_metadata field text_token_count
    dataset = dataset.map(add_token_count, with_indices=True)

    # Calculate total time taken
    total_time = time.time() - start_time
    print(f"Embeddings Generated: {len(dataset)} chunks")
    print(f"Total Time taken: {total_time:.2f} seconds\n")

    return dataset

def save_dataset_to_local(dataset, folder_name='hf_dataset'):

    # Get the current directory of the script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the path to the 'dataset' folder
    dataset_directory = os.path.join(current_dir, folder_name)

    # Conditionally create the 'dataset' folder if it doesn't exist
    if not os.path.exists(dataset_directory):
        os.makedirs(dataset_directory)
        print(f"Created new folder: {dataset_directory}")
    else:
        print(f"Folder already exists: {dataset_directory}")

    # Save the dataset to the 'dataset' folder
    dataset.save_to_disk(dataset_directory)
    print(f"Dataset saved to {dataset_directory}")

    return dataset_directory


# UNUSED / LEGACY FUNCTIONS:

def initialize_model_and_tokenizer_legacy(model_name, device='cpu'):
    """Initialize and return the tokenizer and model."""
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False, trust_remote_code=True)
    
    # Move model to input device
    model.to(device)

    print(f"\n{model_name} embedding model loaded to {device}\n")
    
    return tokenizer, model

def get_dataset_embeddings_legacy(dataset: Dataset, batch_size: int, tokenizer, model, device: str, text_column: str = "markdown_text", use_plaintext: bool = True):
    all_embeddings = []
    all_token_counts = []

    # Start tracking time for the entire embedding process
    start_time = time.time()

    for start_idx in range(0, len(dataset), batch_size):
        end_idx = min(start_idx + batch_size, len(dataset))
        batch = dataset[start_idx:end_idx]
        batch_texts = batch[text_column]

        if use_plaintext:
            batch_texts = [markdown_to_text(text) for text in batch[text_column]]

        print(f"Processing batch indicies: {start_idx} to {end_idx - 1}")

        # Tokenize and move to device
        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=8192)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        token_counts = tokens["attention_mask"].sum(dim=1).tolist() # Attention mask is 0 for padding, 1 for actual tokens

        # Get embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        # Convert to list of vectors and store
        all_embeddings.extend(embeddings.cpu().numpy())
        all_token_counts.extend(token_counts)

    # Add embeddings as new column to the dataset
    dataset = dataset.add_column("embedding", all_embeddings)

    # Modify chunk_metadata in-place to add token_count
    def add_token_count(example, idx):
        example["chunk_metadata"]["text_token_count"] = all_token_counts[idx]
        return example

    # Use map with index to update each example's chunk_metadata field text_token_count
    dataset = dataset.map(add_token_count, with_indices=True)

    # Calculate total time taken
    total_time = time.time() - start_time
    print(f"Embeddings Generated: {len(dataset)} chunks")
    print(f"Total Time taken: {total_time:.2f} seconds")

    return dataset