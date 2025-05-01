# Configuration and argument parsing setup
import argparse

def setup_argparser():
    """
    Sets up the argument parser for command-line arguments.

    Returns:
    - argparse.ArgumentParser: The configured ArgumentParser object.
    """
    parser = argparse.ArgumentParser(description="Process and upload ArXiv papers and metadata.")
    
    # Add arguments for other configurable variables as needed
    parser.add_argument('--pdf_folderpath', type=str, default='arxiv_pdfs',
                        help='Path to the folder containing arXiv PDFs')
    parser.add_argument('--json_folderpath', type=str, default='json_metadata',
                        help='Path to the folder containing JSON metadata')
    parser.add_argument('--get_images', type=bool, default=True,
                        help='Path to save images')
    parser.add_argument('--image_path', type=str, default='images',
                        help='Path to save images')
    parser.add_argument('--hf_login_key', type=str, default='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',
                        help='Hugging Face API key')
    parser.add_argument('--hf_text_repo', type=str, default="JohnVitz/NLP_Final_Project_ArXiv_Parsed2",
                        help='Hugging Face text repository name')
    parser.add_argument('--hf_image_repo', type=str, default="JohnVitz/NLP_Final_Project_ArXiv_Parsed_Images2",
                        help='Hugging Face image repository name')
    parser.add_argument('--pdf_zip_path', type=str, default='raw_data_sources/compiled_eligible_cited_arxiv_pdfs.zip',
                        help='Path to the PDF zip file')
    parser.add_argument('--metadata_zip_path', type=str, default='raw_data_sources/cited_paper_metadata.zip',
                        help='Path to the metadata zip file')
    
    return parser

def parse_args():
    """
    Parses the command-line arguments.

    Returns:
    - argparse.Namespace: The namespace containing the parsed arguments.
    """
    parser = setup_argparser()
    args = parser.parse_args()
    return args
