# Standard library imports
import os
import pickle
from config import parse_args

# General Utility Functions
from get_chunks.raw_data_extraction_utils import extract_pdfs
from get_chunks.raw_data_extraction_utils import parse_and_chunk_pdfs

# Parse command-line arguments
args = parse_args()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # script path
PDF_FOLDERPATH = os.path.join(SCRIPT_DIR, args.pdf_folderpath) # e.g. raw_pdfs
JSON_FOLDERPATH = os.path.join(SCRIPT_DIR, args.json_folderpath) # e.g. json_metadata
IMAGE_PATH = os.path.join(SCRIPT_DIR, args.image_path) # e.g. images
PDF_ZIP_PATH = os.path.join(SCRIPT_DIR, args.pdf_zip_path) # default 'miscellaneous/compiled_eligible_cited_arxiv_pdfs.zip'
METADATA_ZIP_PATH = os.path.join(SCRIPT_DIR, args.metadata_zip_path) # default 'miscellaneous/cited_paper_metadata.zip'
GET_IMAGES = args.get_images # default True

# Function to extract and parse data
def main():
    # Ensure directories exist
    os.makedirs(PDF_FOLDERPATH, exist_ok=True)
    os.makedirs(JSON_FOLDERPATH, exist_ok=True)
    os.makedirs(IMAGE_PATH, exist_ok=True)

    # Extract pdfs from zip
    extract_pdfs(
        mkdirs=True, 
        pdf_folderpath=PDF_FOLDERPATH, 
        pdf_zip_path=PDF_ZIP_PATH, 
    )
    
    # Parse and chunk PDFs
    parsed_pdf_data = parse_and_chunk_pdfs(
        pdf_folderpath=PDF_FOLDERPATH,
        script_dir=SCRIPT_DIR,
        page_chunks=False,
        table_strategy="lines_strict",
        write_images=GET_IMAGES,
        image_path=IMAGE_PATH,
        image_format="jpg",
        image_size_limit=0.05,
        dpi=150
    )

    RAW_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "raw_parsed_pdfs.pkl")
    print(RAW_OUTPUT_PATH)

    with open(RAW_OUTPUT_PATH, "wb") as f:
        pickle.dump(parsed_pdf_data, f)

if __name__ == "__main__":
    main()
