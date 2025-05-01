# Standard Libraries
import os
import re
import itertools
import json

# HF Dataset
from datasets import Dataset, Value, Features, Sequence

# %pip install -qU langchain-text-splitters
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

def add_metadata(parsed_pdfs_dict={}, input_metadata_dict={}):
    """
    Adds metadata to parsed PDFs based on the provided metadata dictionary.

    Parameters:
        parsed_pdfs_dict (dict): A dictionary containing parsed PDF data.
        input_metadata_dict (dict): A dictionary containing metadata for the PDFs.

    Returns:
        dict: The updated dictionary with added metadata.
    """
    pdfs_keys_list = list(parsed_pdfs_dict.keys())
    metadata_keys_list = list(input_metadata_dict.keys())

    missed_pdfs = set(pdfs_keys_list) - set(metadata_keys_list)
    missed_metadata_list = set(metadata_keys_list) - set(pdfs_keys_list)

    if len(missed_pdfs) > 0:
        print(f"Warning - Missed pdf ids: {missed_pdfs}")

    if len(missed_metadata_list) > 0:
        print(f"Warning - Missed metadata ids: {missed_metadata_list}")

    overlapping_arxiv_ids = list(set(metadata_keys_list) & set(pdfs_keys_list))

    for arxiv_id in overlapping_arxiv_ids:

        arxiv_id_metadata = input_metadata_dict[f"{arxiv_id}"]

        parsed_pdfs_dict[f'{arxiv_id}'] = {'markdown' : parsed_pdfs_dict[f'{arxiv_id}'],
                                           'pdf_metadata' : {'id' : arxiv_id_metadata['id'],
                                                             'title' : arxiv_id_metadata['title'],
                                                             'categories' : arxiv_id_metadata['categories'].split(' ')
                                                             }
                                           }

    return parsed_pdfs_dict

def pdf_splitter(input_markdown_and_metadata={}, chunk_size=2500, chunk_overlap=250):
    """
    Splits a markdown document into smaller chunks based on headers and character size.

    Parameters:
        input_markdown_and_metadata (dict): A dictionary containing the markdown text and its metadata.
        chunk_size (int): The maximum number of characters in each chunk.
        chunk_overlap (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of dictionaries, each containing a chunk of the markdown text and its metadata.
    """
    markdown_document = input_markdown_and_metadata['markdown']
    pdf_metadata = input_markdown_and_metadata['pdf_metadata']

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
        ("#####", "Header 5")
    ]

    # Markdown Header split for each input markdown page chunk
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on, strip_headers=False
    )
    md_header_splits = markdown_splitter.split_text(markdown_document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        #separators=["\n\n", ". ", "? ", "! ", ", ", " ", ""],  # Prioritizing sentence boundaries
    )

    # Character Split inside of each header chunk
    recursive_re_split = text_splitter.split_documents(md_header_splits)

    final_page_chunked = []

    for header_chunk_subset in recursive_re_split:

        header_chunk_markdown_text = header_chunk_subset.page_content
        header_chunk_metadata = header_chunk_subset.metadata

        new_chunk = {'markdown_text' : header_chunk_markdown_text,
                    'pdf_metadata' : pdf_metadata,
                    'header_metadata' : header_chunk_metadata,
                    }

        final_page_chunked.append(new_chunk)

    return final_page_chunked

def test_chunking_method(all_pdfs, chunk_size=2500, chunk_overlap=250):
    """
    Re-chunks the pages of PDFs into smaller chunks.

    Parameters:
    - all_pdfs (dict): A dictionary where keys are identifiers and values are PDF content.
    - chunk_size (int): The size of each chunk. Default is 2500.
    - chunk_overlap (int): The overlap between consecutive chunks. Default is 250.

    Returns:
    - list: A list of new chunks grouped by page.
    """
    new_chunks = []
    my_keys = list(all_pdfs.keys())

    for key in my_keys:

        granular_chunks = pdf_splitter(all_pdfs[f'{key}'],
                                        chunk_size=chunk_size,
                                        chunk_overlap=chunk_overlap) # re-chunk the existing page chunks

        new_chunks.append(granular_chunks)

    # Flatten the chunks
    my_flattened_chunks = list(itertools.chain(*new_chunks))

    return my_flattened_chunks # this returns the new chunks from the given document in group by pdf

# This function checks if a header is a 'references' header
def is_references_header(header: str) -> bool:
    """
    Checks if a header text is a 'references' header.

    Parameters:
    - header (str): The header text to check.

    Returns:
    - bool: True if the header is a 'references' header, False otherwise.
    """
    # Remove formatting (e.g., asterisks), extra spaces, and lowercase it
    cleaned = re.sub(r'[^a-zA-Z]', '', header).lower()
    return cleaned == 'references'# or cleaned == 'reference'

# Chunks have to be sequentially ordered according to title for this to work (this is default behavior of parsing)
def get_chunk_metadata(chunk_list=[]):
    """
    Adds metadata to each chunk indicating its type.

    Parameters:
    - chunk_list (list): A list of chunks with their respective metadata.

    Returns:
    - list: The updated list of chunks with added metadata.
    """
    last_title = ''
    chunk_type = 'body'

    new_chunk_list = []

    for i, chunk in enumerate(chunk_list):
        new_chunk_list.append(chunk)

        current_title = chunk['pdf_metadata']['title']

        if last_title != current_title:
          #print(i, chunk)
            last_title = current_title

            chunk_type = 'title'

        else:
            test_header_text = False
            chunk_headers_list = list(chunk['header_metadata'].values())

            if len(chunk_headers_list) > 1: # if there is more than one header
                for header_text in chunk_headers_list: # for each key_value_pair
                    test_header_text = is_references_header(header_text) # check each header's text

                    if test_header_text == True:
                        break
                    
                    ## Add a function that looks at full text for character / structure level detection of "references" sections instead of only testing our recorded headers

            if test_header_text:
                chunk_type = 'references'
            else:
                chunk_type = 'body'

        new_chunk_list[i]['chunk_metadata'] = {'chunk_type' : chunk_type}

    return chunk_list

def upload_dataset_from_dict_list(input_parsed_dict_list=[]):
    """
    Uploads a dataset from a list of parsed dictionaries.

    Parameters:
    - input_parsed_dict_list (list): A list of dictionaries containing parsed data.

    Returns:
    - Dataset: A Hugging Face Dataset object with the specified features.
    """
    dataset_rows = []

    # This step is uncessary because the dataset is already in the proper format, this is just here for clarity on structure
    #for chunk in input_parsed_dict_list:
        #row = {
        #    'markdown_text': chunk['markdown_text'], # The actual text of the chunk in markdown format
        #    'pdf_metadata': chunk['pdf_metadata'],  # The metadata of the pdf of the chunk
        #    'header_metadata': chunk['header_metadata'],  # The metadata of the headers of the chunk
        #    'chunk_metadata' : chunk['chunk_metadata'] # The chunk specific metadata
        #}

        #dataset_rows.append(row)

    features = Features({
        'markdown_text': Value('string'),  # The content of the markdown file (text)
        'pdf_metadata' : {
            'id' : Value('string'), # ID of arxiv paper
            'title' : Value('string'), # Title of paper
            'categories' : Sequence(Value('string')), # List of categories of the PDF that the given chunk originates from
        },
        'header_metadata' : {
            "Header 1" : Value('string'),
            "Header 2" : Value('string'),
            "Header 3" : Value('string'),
            "Header 4" : Value('string'),
            "Header 5" : Value('string')
        },
        'chunk_metadata' : {
            'chunk_type' : Value('string')
        }
    })

    # Create the Hugging Face Dataset with specified features
    dataset = Dataset.from_list(input_parsed_dict_list, features=features)

    return dataset

def save_intial_chunks_dataset_locally(dataset: Dataset, config: dict = {}, save_path: str = "./local_chunk_dataset"):
    """
    Saves a Hugging Face dataset to a local folder in dataset format, and records chunking config.

    Parameters:
    - dataset (Dataset): The Hugging Face dataset to save.
    - save_path (str): Local directory path where the dataset will be saved.
    - chunk_size (int): The chunk character length used for generation.
    - chunk_overlap (int): The character overlap used between chunks.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"📁 Created directory: {save_path}")
    else:
        print(f"📁 Saving to existing directory: {save_path}")

    # Save dataset
    dataset.save_to_disk(save_path)
    print(f"✅ Dataset saved to: {save_path}")

    # Save config
    config_path = os.path.join(save_path, "chunk_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"📝 Chunk config saved to: {config_path}")
