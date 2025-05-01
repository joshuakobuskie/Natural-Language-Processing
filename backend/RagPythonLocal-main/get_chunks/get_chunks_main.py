# Standard library imports
import os
import pickle
from config import parse_args
from datetime import datetime

# Hugging Face Login
from huggingface_hub import login

# General Utility Functions
from raw_data_extraction_utils import extract_metadata
from raw_data_extraction_utils import metadata_from_json

# Image Chunking Functions
from image_utils import create_image_dataset

# Text Chunking Functions
from text_utils import add_metadata
from text_utils import test_chunking_method
from text_utils import get_chunk_metadata
from text_utils import upload_dataset_from_dict_list
from text_utils import save_intial_chunks_dataset_locally

# Parse command-line arguments
args = parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FOLDERPATH = os.path.join(SCRIPT_DIR, args.json_folderpath)       # e.g. json_metadata
IMAGE_PATH = os.path.join(SCRIPT_DIR, args.image_path) 
HF_LOGIN_KEY = args.hf_login_key # default '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'
HF_TEXT_REPO = args.hf_text_repo # default "JohnVitz/NLP_Final_Project_ArXiv_Parsed"
HF_IMAGE_REPO = args.hf_image_repo # default "JohnVitz/NLP_Final_Project_ArXiv_Parsed_Images"
PDF_ZIP_PATH = os.path.join(SCRIPT_DIR, args.pdf_zip_path) # default 'miscellaneous/compiled_eligible_cited_arxiv_pdfs.zip'
METADATA_ZIP_PATH = os.path.join(SCRIPT_DIR, args.metadata_zip_path) # default 'miscellaneous/cited_paper_metadata.zip'
PARSED_PDF_FILENAME = "raw_parsed_pdfs.pkl"
LOCAL_PARSED_PDF_PATH = os.path.join(SCRIPT_DIR, PARSED_PDF_FILENAME)

PUSH_TO_HUB = False
IMAGES = args.get_images

CHUNK_CHAR_LENGTH = 2500
CHUNK_OVERLAP = 250

config_metadata = {
    "chunk_char_length": CHUNK_CHAR_LENGTH,
    "chunk_overlap": CHUNK_OVERLAP,
    "filter_applied": False,  # or a list of filters applied
    "get_images": IMAGES,
    "push_to_hub": PUSH_TO_HUB,
    "hf_text_repo": HF_TEXT_REPO,
    "hf_image_repo": HF_IMAGE_REPO if IMAGES else None,
    "dataset_save_time": datetime.now().isoformat(),
    "source": LOCAL_PARSED_PDF_PATH,
}

# Function to extract and parse data
def main():
    # Load Parsed Output from `.pkl` file
    with open(LOCAL_PARSED_PDF_PATH, "rb") as f:
        parsed_pdf_data = pickle.load(f)
    
    # Load metadata
    extract_metadata(
        metadata_zip_path=METADATA_ZIP_PATH,
        json_folderpath=JSON_FOLDERPATH,
        mkdirs=True,
    )
    my_metadata = metadata_from_json(JSON_FOLDERPATH)
    
    # Login to Hugging Face Hub
    login(HF_LOGIN_KEY)
    
    # Add metadata to parsed PDF data
    pdf_dict_with_metadata = add_metadata(parsed_pdf_data, my_metadata)
    
    # Test chunking method
    my_chunks = test_chunking_method(pdf_dict_with_metadata, chunk_size=CHUNK_CHAR_LENGTH, chunk_overlap=CHUNK_OVERLAP)
    
    # Get chunk metadata
    final_chunk_list = get_chunk_metadata(my_chunks)

    # Upload text dataset to Hugging Face Hub
    my_text_dataset = upload_dataset_from_dict_list(final_chunk_list)

    # Save Dataset Locally
    #folder_name = f"hf_dataset_chunksize{CHUNK_CHAR_LENGTH}_overlap{CHUNK_OVERLAP}"
    #save_intial_chunks_dataset_locally(dataset=my_text_dataset, folder_name=folder_name)
    save_folder = f"./local_chunk_dataset/chunk{CHUNK_CHAR_LENGTH}_overlap{CHUNK_OVERLAP}"
    save_intial_chunks_dataset_locally(
        dataset=my_text_dataset,
        config=config_metadata,
        save_path=save_folder,
    )

    if IMAGES:
        # Create and upload image dataset to Hugging Face Hub
        my_image_dataset = create_image_dataset(IMAGE_PATH)

    # Push to Hub conditionally

    if PUSH_TO_HUB:
        my_text_dataset.push_to_hub(HF_TEXT_REPO)

        if IMAGES:
            my_image_dataset.push_to_hub(HF_IMAGE_REPO)

if __name__ == "__main__":
    main()
