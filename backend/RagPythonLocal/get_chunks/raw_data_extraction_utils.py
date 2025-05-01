import os
import re
import json
import zipfile
import pymupdf4llm
import time

# Miscellaneous
from tqdm import tqdm  # Make sure tqdm is installed: pip install tqdm

def extract_pdfs(pdf_zip_path, pdf_folderpath, mkdirs=True):
    """
    Extracts PDF files from a zip archive into a specified directory.

    Parameters:
        pdf_zip_path (str): The path to the zip file containing PDFs.
        pdf_folderpath (str): The directory path where PDF files will be extracted.
        mkdirs (bool): If True, create the target directory if it does not exist.

    Returns:
        None
    """
    # Ensure the PDF ZIP path is absolute
    if not os.path.isabs(pdf_zip_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_zip_path = os.path.join(script_dir, pdf_zip_path)

    # Check if the ZIP file exists
    if not (os.path.exists(pdf_zip_path) and os.path.isfile(pdf_zip_path)):
        raise FileNotFoundError(f"PDF ZIP file not found at: {pdf_zip_path}")

    try:
        if mkdirs:
            os.makedirs(pdf_folderpath, exist_ok=True)

        # Unzip PDFs
        print(f"Extracting PDFs from: {pdf_zip_path} to {pdf_folderpath}")
        with zipfile.ZipFile(pdf_zip_path, 'r') as zip_ref:
            zip_ref.extractall(pdf_folderpath)

    except Exception as e:
        raise RuntimeError(f"Failed to extract PDFs. Error details: {str(e)}")

    print("✅ PDFs extracted to:", pdf_folderpath)

def extract_metadata(metadata_zip_path, json_folderpath, mkdirs=True):
    """
    Extracts metadata from a zip archive into the specified directory.

    Parameters:
        metadata_zip_path (str): The path to the zip file containing metadata.
        json_folderpath (str): The directory path where JSON metadata files will be extracted.
        mkdirs (bool): If True, create the target directory if it does not exist.

    Returns:
        None
    """
    # Ensure that the ZIP path is absolute
    if not os.path.isabs(metadata_zip_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_zip_path = os.path.join(script_dir, metadata_zip_path)
    
    # Check if the ZIP file exists
    if not (os.path.exists(metadata_zip_path) and os.path.isfile(metadata_zip_path)):
        raise FileNotFoundError(f"Metadata ZIP file not found at: {metadata_zip_path}")

    try:
        if mkdirs:
            os.makedirs(json_folderpath, exist_ok=True)
        
        # Unzip metadata
        print(f"Extracting metadata from: {metadata_zip_path} to {json_folderpath}")
        with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
            zip_ref.extractall(json_folderpath)

    except Exception as e:
        raise RuntimeError(f"Failed to extract metadata. Error details: {str(e)}")

    print("✅ Metadata extracted to:", json_folderpath)

def metadata_from_json(folder_path):
    """
    Reads all JSON files in a specified directory and returns their contents as a dictionary.

    Parameters:
        folder_path (str): The path to the directory containing JSON metadata files.

    Returns:
        dict: A dictionary where keys are filenames (without extension) and values are their corresponding metadata.
    """
    # Directory containing markdown and metadata files
    #os.makedirs(folder_path, exist_ok=True)

    # Initialize a dictionary to hold data by file name
    metadata_data = {}

    # Read all JSON metadata files to get metadata
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  # Process metadata files
            #print('dsafasd')
            metadata_filepath = folder_path + '/' + filename

            # Read JSON metadata content
            with open(metadata_filepath, 'r') as meta_file:
                metadata = json.load(meta_file)
                #print(metadata)

            # Store metadata by the base filename (without the .json extension)
            metadata_data[os.path.splitext(filename)[0]] = metadata

    return metadata_data

def parse_and_chunk_pdfs(pdf_folderpath='', script_dir='', **kwargs):
    """
    Parses PDF files in a specified directory and chunks their content using pymupdf4llm.

    Parameters:
        pdf_folderpath (str): The path to the directory containing PDF files.
        script_dir : The path of the script being executed
        **kwargs: Additional keyword arguments passed to pymupdf4llm.to_markdown.

    Returns:
        dict: A dictionary where keys are filenames (without extension) and values are their corresponding Markdown content.
    """
    md_text_images_dict = {}
    fail_ct = 0
    slow_files = []  # List to keep track of PDFs that took longer than 20 seconds

    pdf_files = sorted([f for f in os.listdir(pdf_folderpath) if f.endswith(".pdf")])

    for filename in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        full_path = os.path.join(pdf_folderpath, filename)

        #print(full_path)
        start_time = time.time()  # Record the start time

        try:
            # Try converting PDF to markdown using pymupdf4llm
            md_text_images = pymupdf4llm.to_markdown(
                doc=full_path,
                **kwargs
            )

            end_time = time.time()  # Record the end time
            elapsed_time = end_time - start_time

            if elapsed_time > 20:
                slow_files.append((filename, elapsed_time))  # Add to slow files list

            # Clean filename to get ID (e.g., remove versioning like v1.pdf)
            edited_filename = re.sub(r'v\d+\.pdf$', '', filename)
            md_text_images_dict[edited_filename] = md_text_images

        except Exception as e:
            print(f"❌ Failed to process {filename}: {e}")
            fail_ct += 1
        
        #break # for debugging / testing, remove to get all pdfs parsed

    # Save slow files data to a .txt file
    if slow_files:
        slow_files_path = os.path.join(script_dir, "slow_pdf_processing_times.txt")

        with open(slow_files_path, "w") as f:
            for file_info in slow_files:
                line = f"{file_info[0]} took {file_info[1]:.2f} seconds\n"
                f.write(line)
        
        # Print the contents of the .txt file
        with open(slow_files_path, "r") as f:
            print("\n🔥 Contents of slow_pdf_processing_times.txt:")
            print(f.read())

    print(f"🚨 Total failed files: {fail_ct}\n")

    return md_text_images_dict

# UNUSED / LEGACY

def extract_zips_and_json(mkdirs=True, pdf_folderpath=str, json_folderpath=str, pdf_zip_path=str, metadata_zip_path=str):
    """
    Extracts PDFs and metadata from zip archives into specified directories.

    Parameters:
        mkdirs (bool): If True, create the target directories if they do not exist.
        pdf_folderpath (str): The directory path where PDF files will be extracted.
        json_folderpath (str): The directory path where JSON metadata files will be extracted.
        pdf_zip_path (str): The path to the zip file containing PDFs.
        metadata_zip_path (str): The path to the zip file containing metadata.

    Returns:
        None
    """
    # Ensure that the ZIP paths are absolute
    if not os.path.isabs(pdf_zip_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_zip_path = os.path.join(script_dir, pdf_zip_path)
    
    if not os.path.isabs(metadata_zip_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        metadata_zip_path = os.path.join(script_dir, metadata_zip_path)
    
    # Check if the ZIP files exist
    if not (os.path.exists(pdf_zip_path) and os.path.isfile(pdf_zip_path)):
        raise FileNotFoundError(f"PDF ZIP file not found at: {pdf_zip_path}")
    
    if not (os.path.exists(metadata_zip_path) and os.path.isfile(metadata_zip_path)):
        raise FileNotFoundError(f"Metadata ZIP file not found at: {metadata_zip_path}")

    try:
        if mkdirs:
            os.makedirs(pdf_folderpath, exist_ok=True)
            os.makedirs(json_folderpath, exist_ok=True)
        
        # Unzip PDFs
        print(f"Extracting PDFs from: {pdf_zip_path} to {pdf_folderpath}")
        with zipfile.ZipFile(pdf_zip_path, 'r') as zip_ref:
            zip_ref.extractall(pdf_folderpath)
        
        # Unzip metadata
        print(f"Extracting metadata from: {metadata_zip_path} to {json_folderpath}")
        with zipfile.ZipFile(metadata_zip_path, 'r') as zip_ref:
            zip_ref.extractall(json_folderpath)
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract files. Error details: {str(e)}")

    print("✅ PDFs extracted to:", pdf_folderpath)
    print("✅ Metadata extracted to:", json_folderpath)