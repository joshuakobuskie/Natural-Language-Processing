# Sections:

1. How to run

2. Necessary Improvements

3. Developing / Utilizing a Test Dataset and Evaluating with RAGAS Framework

4. Extra

<br>

# 1. How to run

## NEEDED FOR ALL STEPS AFTER THIS:

You need the following:

- a python virtual environment (my version is 3.12.3)

- activate the virtual environment

- install `requirements.txt`

After, here is how to get individual files to run

<br>

## For `user_response_main.py`

- move the file `user_response_main.py` to the same level as the virtual environment

- create a folder named `qdrant_vector_store` at the same level as the `user_response_main.py` file

- place the qdrant_vectors.pkl inside of the folder (you can get that here: https://drive.google.com/file/d/1ktWUzwCrf4u28ZOkI6pJRwlq1l_ndqsx/view?usp=drive_link)

and then just type:

`python user_response_main.py` 

into the terminal.

<br>

## For `/get_chunks/*.py` (all python files in get_chunks folder)

NOTE: IGNORE get_top_papers.py

`get_chunks_main.py` is the main file at this folder level. You have to have `cited_paper_metadata.zip` and `compiled_eligible_cited_arxiv_pdfs.zip` placed in the `raw_data_sources` folder in order to run `get_chunks_main.py`.

IMPORTANT: PLEASE CHECK `config.py` TO EXAMINE THE VARIOUS ARGUMENTS THAT CAN BE PASSED TO `get_chunks_main.py`. 
IMPORTANT: PLEASE SET UP YOUR OWN HUGGINGFACE ACCOUNT AND USE YOUR OWN LOGIN KEY / YOUR OWN REPO NAMES BEFORE TESTING THIS FUNCTION OUT. AS IT STANDS, BY DEFAULT IT WILL DELETE AND REPLACE MY DATASET WITH WHATEVER YOU GENERATE.

The following are arguments for `get_chunks_main.py` taken from `config.py` in the same `get_chunks` folder:

```python
    # Add arguments for other configurable variables as needed
    parser.add_argument('--pdf_folderpath', type=str, default='raw_pdfs',
                        help='Path to the folder containing PDFs')
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
    parser.add_argument('--pdf_zip_path', type=str, default='miscellaneous/compiled_eligible_cited_arxiv_pdfs.zip',
                        help='Path to the PDF zip file')
    parser.add_argument('--metadata_zip_path', type=str, default='miscellaneous/cited_paper_metadata.zip',
                        help='Path to the metadata zip file')
```
IMPORTANT: PLEASE CHANGE `hf_login_key`, `hf_text_repo`, `hf_image_repo` TO UPLOAD YOUR TEST CHUNKS

IMPORTANT: IT WOULD BE GOOD TO SIMPLY PARSE ALL OF THE DATA AND SAVE IT PARSED BEFORE CHUNKING SO THAT WE DO NOT HAVE TO PARSE EVERY TIME THAT WE WANT TO RE-CHUNK.

IMPORTANT: SOME OF THE PARAMETERS MAY CAUSE ERRORS WHEN CHANGED (ALMOST CERTAINLY `get_images = False` WILL ERROR BECAUSE `if` STATEMENT HAS NOT BEEN IMPLEMENTED YET FOR SELECTIVE CREATION OF IMAGE DATASET). THIS WOULD BE GOOD TO ADD BECAUSE CHANGES TO THE TEXT CHUNKING PROCESS HAVE NO EFFECT ON EXTRACTED IMAGES WHICH MEANS WE DO NOT NEED THEM TO BE EXTRACTED EACH TIME.

<br>

## For `/qdrant_vector_store/*.py`

Running `qdrant_vector_store/get_embeddings_main.py` is a bottleneck for those of us without access to local GPU. First off, to run, you need to pass your own `HF_TEXT_REPO` at the beginning of the file

```python
# Get the Dataset from HuggingFace
HF_TEXT_REPO = "JohnVitz/NLP_Final_Project_ArXiv_Parsed2"
dataset_from_HF = load_dataset(HF_TEXT_REPO)
```

This essentially requires a GPU in order to be run efficiently; you are better off simply copying the following colab notebook:

https://colab.research.google.com/drive/1TNbgb1aYCk1YGJjJ1D5XpONi6qo-QlBP?usp=sharing

and using it to perform chunking with colab. You will have to manually change the `HF_TEXT_REPO` path here to your own (at the top of the file).

This step is necessary to test the performance of any changes to chunking methods during the actual response generation phase.

<br>

# 2. Necessary Improvements:

The following section will list necessary improvements needed for this project. I will include my estimations of the challenge for each task.

Three Categories for status of listed tasks:

- (OPEN)        : Open tasks
- (IN PROGRESS) : In - progress tasks
- (DONE)        : Finished tasks

## Code Health

<br>

### (OPEN) Code Cleaning / Refactoring - `RagPythonLocal/*.py`

Cleaning all python code up and refactoring functions as well as variables, adding doc strings to each function and making everything more straightforward to understand, so that if we need to make changes we have a better idea of what we are actually changing. Generally trying to improve the interpretability of everything that we are doing, explaining anything needs an explanation.

Challenge: probably very time consuming

### (DONE) Seperation of Initial Parsing Step from Chunking Step - `/get_chunks/*.py` (specifically splitting `get_chunks_main.py` into `parse_pdfs_main.py` and `get_chunks_main.py`)

We should probably disconnect the Parsing Step from the Chunking Step. If our selected `.pdf` files for our knowledge base does not change, we do not need to parse all of the `.pdf` files over again in order to test different chunking methods. So, the parsing and chunking should be un-coupled and we should save the results from parsing immediately in a format we can then load when testing / utilizing our chunking method. I propose we move this process to another file `parse_pdfs_main.py` and change `get_chunks_main.py` so that instead of requiring a call to our parsing method to start chunking, we can utilize our saved parsed text instead. Just keep in mind that at this step we need to preserve whatever the required input structure is for the next phase at `get_chunks_main.py`.

Challenge: should be easy, just a matter of splitting files

Summary:

Parsing step: `parse_pdfs_main.py`
Chunking step: `get_chunks_main.py`

The steps are now de-coupled from each other. Running `parse_pdfs_main.py` should only be done if our pdf subset changes, and saves the chunked output to `raw_parsed_pdfs.pkl` in the same folder. Running `get_chunks_main.py` will load `raw_parsed_pdfs.pkl` and add metadata to push the dataset to huggingface.

<br>

## Front End Integration and Utilization of History:

### (OPEN) Integration with Front End - `user_response_main.py`

Integrating the `user_response_main.py` to a front end that allows the file to run continuously is important. We need the embedding model to remain loaded in memory because it is needed for each query in order to perform our chunk retrieval, and loading it each time we input a new query would be very costly in terms of latency between user query inputs and getting actual LLM outputs. Allowing our back end response generator to run continouously will also allow us to store history in the back end, which we can utilize for more advanced history utilization methods such as getting similarity between our prior query embeddings / prior chunk embeddings and current query embeddings to decide what history to use.

Challenge: time consuming, reliant on one person's knowledge (sorry josh :( we don't know front end))

### (DONE) Utilizing Chat History - `user_response_main.py`

Adding a history component that is usable by the model - for this, there needs to be variable(s) keeping track of the number of interactions the user has with the LLM in the session. Initially we can leave it as naive where we just look back some # of steps and pass it as history (we need to look into gemini API for this), however we might be able to get better performance by doing semantic comparison between current and prior queries / current query and prior chunks.

Challenge: there are many different ways to do this, so we need to decide the best way before implementing it (if we want to implement more advanced history searching, this should be done in another qdrant collection)

<br>

## Better Chunk Filtering / Analysis:

### (DONE) Chunk Type Detection Improvements - `/get_chunks/*.py` (all python files in get_chunks folder)
- Try to detect references better (is a chunk's text 'title, 'body', or 'references'? store result in chunk_metadata['chunk_type']) and specifically try to improve detection / removal of chunks that consist of references (this may be hard, need to brainstorm on this)

### (IN PROGRESS) Chunk Character Specific Filtering - `/get_chunks/*.py` (all python files in get_chunks folder)
- Try to detect chunks that consist of basically nonsense. For example, I saw a chunk while messing around that consists entirely of 'G', punctuation, and blank space characters. If we broke up a chunk like this by character and measured the length of the set, it would be like ~4 characters total. So maybe doing something like this could be advantageous to try to detect bad chunks.

### (DONE) Chunk Token Count Filtering - `qdrant_vector_store/*.py`
- Filter off of minimum token count, which we have in the payload of the Q-drant vectors (so this should be easy, we can just set a minimum bound of ~50 to ~250)

Challenge: Either filter at the end of the embedding generation process, or the beginning of the response generation process. Or create a step in the middle to do chunk analysis and filter in this step. Details on chunk analysis in next subsection.

Summary:

There is a token filter for our points that is applied in `get_embeddings_main.py` function `main()` before upserting to qdrant store to save in qdrant format for export to `qdrant_vectors.pkl`.

```python
# Apply plain text filter `text_token_count` >= 50
MIN_TOKENS = 50
knowledge_base_body_chunks = knowledge_base_body_chunks.filter(lambda chunk: chunk['chunk_metadata']['text_token_count'] >= MIN_TOKENS)
```

### (DONE) Chunk Analysis - file needs to be created / developed for addition to `qdrant_vector_store/*.py`, may need to analyze chunks before applying filters also

NOTE: IF YOU DECIDE TO WORK ON THIS, CONTACT ME (JOHN) BECAUSE I DID THIS FOR NLP ASSIGNMENT 3 AND CAN PROVIDE LINKS TO NOTEBOOK CODE SHOWING HOW TO DO IT

This step should be developed, and should probably start out as (and potentially even remain) an `.ipynb` file titled something along the lines of `chunking_method_metrics.ipynb`. Chunk analysis would entail a file that we can pass a hugginface dataset of chunks to and get metrics about the chunks in the dataset as a return. This can be developed in colab because it is a detached way of generating summaries / metrics regarding our various chunking methods, such as average token count per chunk, variation in token count across all chunks, ratio between chunks designated as `chunk_type` = `title` vs `body` vs `references` and other useful metrics. This is also (probably) where we should do category analysis (aka distribution of `pdf_metadata` categories) and wordcloud generation for our presentation.

Challenge: Building a pipeline that allows for plug and play metrics on potentially different chunking methods. It is crucial that our input and output formats between steps stay the same so that we can use this file in a plug and play manner.

<br>

## Summary in Order of Importance 1 -> 2 -> ... :

- Front End: Most important is integrating the continuous running format -> implementing a useable chat history

- Back End: Most important is seperation of parsing and chunking steps -> code review/cleaning/refactoring -> chunk method refining and analysis

<br>

# 3. (IN PROGRESS) Optimizations / Developing and Utilizing a Test Dataset and Evaluating with RAGAS Framework:

Preface: Our default (if we don't have time for this section) can be to get some curated subset of queries and to pick generated responses to queries that show substantial qualitative differences between our no-RAG LLM and RAG LLM responses.

Otherwise:

Once we finish implementing the front and and reach a satisfactory point where we have generally covered what was listed above, we should try to use RAGAS to do performance testing on our RAG system. We should consult with the professor before / during the beginning of this step, as we may be able to get more insight on how to go about doing this effectively. At a baseline, we can evaluate our single model utlizing different chunk retrieval strategies.

RAGAS Link: https://github.com/explodinggradients/ragas

The biggest challenge with this section in general will be the fact that we have to get a dataset that allows us to evaluate performance on either our direct retrieval task or a task that is directly adjacent to ours, otherwise our metrics won't mean anything relative to our actual task.

NOTE: Here are two videos that I found helpful / interesting as an introduction to the concepts underlying RAG evalution:

1. Building Production-Ready RAG Applications: Jerry Liu - https://youtu.be/TRjq7t2Ms5I?si=ipsxjmDi-a0FP-IC
2. AI Agent Evaluation with RAGAS - https://youtu.be/-_52DIIOsCE?si=HlTQ5pDPMlZWTujU
3. Prompt Engineering, RAG, and Fine-tuning: Benefits and When to Use - https://www.youtube.com/watch?v=YVWxbHJakgg

## Embedding Model Analysis

This needs to be looked into a bit further; there are some details in the first two videos shared above. That being said, this would require use to re-embed our chunks in order to test out directly (unless we use a linear adapter for the vectors, which we are NOT going to try unless it is trivially easy).

I personally think we are better off evaluating different LLMs rather than doing this.

## Generation Step Analysis

This component of evaluation involves metrics concerning the generative model (i.e. the LLM) that we use during our RAG powered inference.

### No Context Model (No RAG) verus Context Model (RAG)

If we do manage to get 

### Zero-Shot vs Few-Shot In-context Learning Comparison

This would involve us giving no examples (our current baseline) for expected generation format / content structure versus us giving 3 / 5 / 8 / 10 (not necessarily every increment of that) currated examples as in-context few shot learning prompts before passing our data to LLM apis.

### Baseline Chunk Retrival vs Topic-level 'Chunk Window' Retrieval

We can take our best performing models from trying different models and/or APIs, and then test them using two different methods of retrieval: First, our baseline simple retrieval method where we feed in chunks as context (we already have this). Second, a topic level window retrieval that uses chunk metadata to retrieve context at a topic level to feed to the model (we don't have this yet).

### Try Different Models and/or APIs and compare results (assuming we can get RAGAS to work with our project)

This sounds complicated, but actually is (probably) simple due to the fact that we have a very modular project that allows for swapping of models:

We aren't utilizing history here, so we do not need to do any passing of history during our inference stage (the time consuming stage).

Testing different models with RAG vs without RAG on QA tasks from dataset utilized with RAGAS evaluation, on different models, whether that means sticking with GEMINI api and simply using older smaller models through to the recent / more powerful ones, or even potentially comparing say GEMINI api models with chatGPT api models. The crucial thing here is to get a test dataset for evaluation ASAP so that we can show test results.

<br>

## Summary

This section would definitely improve our final paper / presentation, and is the final 'piece of the puzzle' when it comes to developing this application (given our time constraints). The hardest part for this will be getting a dataset that allows us to perform metric evaluation, and developing an evaluation pipeline.


<br>

# 4. Extra:

## Re-Ranking Model - Utilizing an additional model to re-rank chunks

We could look into utlizing a re-ranking model for context chunks, however we must stay congniscant of the fact that doing this will incurr more overhead in terms of compuation requirements and latency between responses. It is also not necessarily plug and play in the sense that generally, adding these re-ranking models without fine tuning them can end up harming our results or having no effect other than adding latency, which is why it is here under 4. Extra section.

## Citations - Context Chunk Source Transparency / Convenience for User

Response markdown text citations of utilized chunks (citing their arxiv ids): (try this https://github.com/MadryLab/context-cite) that provide the arxiv id and title of papers being cited. 

We can actually directly cite the website link if we get the `arxiv_id` from our chunk's `pdf_metadata`:

```python
arxiv_id = 1701.04862
abstract_link = https://arxiv.org/abs/

web_link = abstract_link + arxiv_id # https://arxiv.org/abs/1701.04862
```

`abstract_link` link is universal while `arxiv_id` is what we would select through context-citation that relates the text our LLM generates back to our chunks
