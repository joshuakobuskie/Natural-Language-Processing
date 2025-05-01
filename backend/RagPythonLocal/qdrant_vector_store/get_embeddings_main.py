import os

# HuggingFace
from datasets import load_dataset
from datasets import load_from_disk

# Torch for cuda availability check
import torch

# Qdrant Client
from qdrant_client import QdrantClient

# Data Utils (embedding generation and dataset formatting)
from data_embedding_utils import instantiate_model_and_tokenizer
from data_embedding_utils import get_dataset_embeddings
from data_embedding_utils import save_dataset_to_local

# Qdrant utils (qdrant upserting and exporting)
from qdrant_utils import upsert_dataset_to_qdrant_client
from qdrant_utils import export_qdrant_vectors
from qdrant_utils import instantiate_collection

def main():
    # Get the Dataset from HuggingFace
    HF_TEXT_REPO = "JohnVitz/NLP_Final_Project_ArXiv_Parsed2"
    dataset_from_HF = load_dataset(HF_TEXT_REPO)
    
    # Define the base directory for embedding models
    QDRANT_STORE_LOCATION = 'qdrant_vector_store'
    LOCAL_BASE_DIR = 'local_embedding_models'
    MODEL_LOCATION = os.path.join(QDRANT_STORE_LOCATION, LOCAL_BASE_DIR)
    MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
    LOCAL_MODEL_DIR = os.path.join(MODEL_LOCATION, MODEL_NAME)
    
    # Get embedding model and tokenizer from HuggingFace or local storage, load it to DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER, MODEL = instantiate_model_and_tokenizer(MODEL_NAME, DEVICE, LOCAL_MODEL_DIR)

    # Define the filter subset of the full dataset to get embeddings for (set to body for valid chunks)
    DATA_SUBSET = 'body'
    knowledge_base_body_chunks = dataset_from_HF['train'].filter(lambda chunk: chunk['chunk_metadata']['chunk_type'] == DATA_SUBSET) # set back to body for valid chunks

    # Get the Embeddings
    knowledge_base_body_chunks = get_dataset_embeddings(
        dataset=knowledge_base_body_chunks,
        batch_size=24,
        tokenizer=TOKENIZER,
        model=MODEL,
        device=DEVICE,
        use_plaintext=True
    )

    # Apply plain text filter `text_token_count` >= 50
    MIN_TOKENS = 50
    knowledge_base_body_chunks = knowledge_base_body_chunks.filter(lambda chunk: chunk['chunk_metadata']['text_token_count'] >= MIN_TOKENS)

    # Save the dataset to a local subfolder in the current folder
    hf_dataset_local_dir = save_dataset_to_local(dataset=knowledge_base_body_chunks, folder_name='hf_dataset')
    local_dataset = load_from_disk(hf_dataset_local_dir)
    
    EMBEDDING_SIZE = len(local_dataset[0]['embedding']) # can also be `MODEL.config.hidden_size`
    #print(embedding_size)

    # Initialize the Qdrant Client (to run either in memory, or through a host i.e. "localhost", through host is untested)
    QDRANT_CLIENT = QdrantClient(location=":memory:")
    
    # Create a Collection that is compatible with the embedding model
    MY_COLLECTION = 'saved_chunks_collection'
    #QDRANT_CLIENT.create_collection(
    #    collection_name=MY_COLLECTION,
    #    vectors_config=VectorParams(size=embedding_size, # Size of Snowflake Embedding Dimensions
    #                                distance=Distance.COSINE), # Cosine similarity for vector search
    #)
    instantiate_collection(qdrant_client=QDRANT_CLIENT, collection_name=MY_COLLECTION, embedding_size=EMBEDDING_SIZE)

    # Upsert the dataset with embeddings to the qdrant client to get format for saving as .pkl
    upsert_dataset_to_qdrant_client(
        dataset=local_dataset,
        collection_name=MY_COLLECTION,
        qdrant_client=QDRANT_CLIENT
    )

    # Export the collection to .pkl format for storage as a file
    export_qdrant_vectors(collection_name=MY_COLLECTION, 
                          qdrant_client=QDRANT_CLIENT, 
                          folder_path='qdrant_vector_store', 
                          output_file="qdrant_vectors.pkl")

if __name__ == '__main__':
    main()
