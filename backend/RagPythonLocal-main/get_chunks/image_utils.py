import os
import re

# Image Display / Loading
from PIL import Image as PILImage
from datasets import Dataset, Value, Features, Image

def parse_image_filenames(image_folder):
    """
    Parses image filenames from a specified directory and organizes them by paper ID and version.

    Parameters:
        image_folder (str): The path to the directory containing image files.

    Returns:
        dict: A dictionary where keys are unique identifiers combining paper ID and version, 
              and values are lists of dictionaries containing image details.
              Each dictionary in the list includes:
                  - 'id': Paper identifier (e.g., "1012.2599")
                  - 'page_number': The page number of the image (int)
                  - 'image_number': The image number on the page (int)
                  - 'full_path': The full path to the image file
    """
    image_data = {}

    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg"):
            # Match filenames like: 1012.2599v1.pdf-11-1.jpg
            match = re.match(r'^(\d+\.\d+)(v\d+)\.pdf-(\d+)-(\d+)\.jpg$', filename)
            if match:
                paper_id, version, page_num, img_num = match.groups()
                key = f"{paper_id}{version}"

                if key not in image_data:
                    image_data[key] = []

                image_data[key].append({
                    "id": paper_id,
                    "page_number": int(page_num),
                    "image_number": int(img_num),
                    "full_path": os.path.join(image_folder, filename)
                })
            else:
                print(f"⚠️ Filename didn't match expected pattern: {filename}")

    return image_data

def create_image_dataset(image_folder):
    """
    Creates a Hugging Face Dataset from images in a specified directory.

    Parameters:
        image_folder (str): The path to the directory containing image files.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object with image data and metadata.
    """
    parsed_image_dict = parse_image_filenames(image_folder)
    rows = []

    for pdf_id, image_list in parsed_image_dict.items():
        for img_info in image_list:
            image_path = img_info['full_path']
            try:
                image = PILImage.open(image_path)

                row = {
                    'image': image,
                    'metadata': {
                        'id': img_info['id'],
                        'image_pdf': f"{pdf_id}.pdf",
                        'image_page_num': img_info['page_number']
                    }
                }

                rows.append(row)
            except Exception as e:
                print(f"❌ Failed to open image {image_path}: {e}")

    features = Features({
        'image': Image(),
        'metadata': {
            'id': Value('string'),
            'image_pdf': Value('string'),
            'image_page_num': Value('int32'),
        }
    })
    # Create the Hugging Face Dataset with specified features
    dataset = Dataset.from_list(rows, features=features)

    return dataset