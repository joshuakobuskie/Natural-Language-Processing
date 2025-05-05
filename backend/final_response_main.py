# Standard Libraries
import os
import time
import numpy as np
from datetime import datetime

# Qdrant_client
from qdrant_client import QdrantClient # Qdrant Vector Database package

# Gemini LLM Output
import google.generativeai as genai

# Embedding Model dependency
import torch

# For loading / using the embedding model and tokenizer
from qdrant_vector_store.data_embedding_utils import instantiate_model_and_tokenizer
from query_embedding_utils import get_query_embedding
from query_embedding_utils import get_context_chunks

# For interacting with Qdrant Client
from qdrant_vector_store.qdrant_utils import load_to_vector_store
from qdrant_vector_store.qdrant_utils import instantiate_collection # initialize collection
from qdrant_vector_store.qdrant_utils import upsert_history_to_vector_store

# For Utilizing BM25 Hybrid Search
from qdrant_vector_store.qdrant_utils import initialize_BM25
import nltk
from rank_bm25 import BM25Okapi

# These functions are used to deal with prompting the LLM / displaying LLM output
from model_prompting_utils import standalone_prompt_model
from model_prompting_utils import build_historical_query_response_thread
from model_prompting_utils import topic_level_context_retrieval
from qdrant_vector_store.data_embedding_utils import markdown_to_text # This function is used for printing the terminal output version of LLM's response for quicker testing
from model_prompting_utils import get_system_prompt
from model_prompting_utils import display_summary

# For filtering
from query_state_filter_utils import current_query_to_prior_queries_filter

# For saving chat output (can load later)
from user_history_utils import save_chat_pkl_by_embedding
from user_history_utils import save_chat_json
from dotenv import load_dotenv

def initialize_gemini_model():
    """
    Loads the API key from the .env file and initializes the Gemini model.
    
    Returns:
    - LLM_MODEL: The initialized Gemini model.
    """
    
    genai.configure()

def main():
    # Define the base directory for embedding models
    QDRANT_STORE_LOCATION = 'qdrant_vector_store'
    LOCAL_BASE_DIR = 'local_embedding_models'
    EMBEDDINGMODEL_LOCATION = os.path.join(QDRANT_STORE_LOCATION, LOCAL_BASE_DIR)
    EMBEDDING_MODEL_NAME = 'Snowflake/snowflake-arctic-embed-l-v2.0'
    LOCAL_MODEL_DIR = os.path.join(EMBEDDINGMODEL_LOCATION, EMBEDDING_MODEL_NAME)

    # Get embedding model and tokenizer from HuggingFace or local storage, load it to DEVICE
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TOKENIZER, EMBEDDING_MODEL = instantiate_model_and_tokenizer(EMBEDDING_MODEL_NAME, DEVICE, LOCAL_MODEL_DIR)
    print()

    # Set up API key
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=GEMINI_API_KEY)

    # Define the model name
    LLM_MODEL_NAME = "gemini-2.0-flash-lite"

    # Define the system prompt
    LLM_SYSTEM_PROMPT = get_system_prompt()

    # Initialize the Gemini model
    LLM_MODEL = genai.GenerativeModel(model_name=LLM_MODEL_NAME, system_instruction=LLM_SYSTEM_PROMPT)

    # Initialize the Qdrant Client and Initial Collection
    QDRANT_CLIENT = QdrantClient(location=":memory:")
    CHUNK_COLLECTION = "search_collection"
    HISTORY_COLLECTION = "history_collection"
    EMBEDDING_SIZE = EMBEDDING_MODEL.config.hidden_size

    # Initialize the qdrant collections to hold our information
    instantiate_collection(qdrant_client=QDRANT_CLIENT, collection_name=CHUNK_COLLECTION, embedding_size=EMBEDDING_SIZE)
    instantiate_collection(qdrant_client=QDRANT_CLIENT, collection_name=HISTORY_COLLECTION, embedding_size=EMBEDDING_SIZE)

    # Load stored knowledge base collection .pkl file
    PKL_FILENAME = 'qdrant_vectors.pkl'
    PKL_STORE_LOCATION = os.path.join(QDRANT_STORE_LOCATION, PKL_FILENAME)
    load_to_vector_store(pkl_filepath=PKL_STORE_LOCATION, qdrant_client=QDRANT_CLIENT, collection_name=CHUNK_COLLECTION)

    # Enable BM25 Hybrid Search
    BM25_SEARCH_FUNCTION = initialize_BM25(PKL_STORE_LOCATION)

    # Enable this to load prior history to states
    USER_QUERY_HISTORY = True

    from user_history_utils import load_chat_pkl

    # Initialize the main storage point of permanent user history information
    if USER_QUERY_HISTORY:
        # load prior history chat from json
        user_query_state_history_embedded = load_chat_pkl("user_output/user_embedded_history.pkl")
        user_query_state_history_non_embedded = load_chat_pkl("user_output/user_non_embedded_history.pkl")

        # Merge both embedded and non-embedded histories
        merged_history = {
            **user_query_state_history_non_embedded,
            **user_query_state_history_embedded
        }

        # Sort by query_number key
        user_query_state_history = dict(sorted(merged_history.items()))

        # Final query count
        #query_num = 0
        query_num = len(user_query_state_history)
        HISTORICAL_QUERY_NUM = len(user_query_state_history)

        if len(user_query_state_history_embedded) > 0:

            upsert_history_to_vector_store(qdrant_client=QDRANT_CLIENT, 
                        collection_name=HISTORY_COLLECTION, 
                        history=user_query_state_history_embedded, # upsert batch of current query, function will filter out queries without embeddings
                        batch=True # upsert the current batch
            )
        
        print(f"Loading chat, start from query number: {query_num}")
    else:

        user_query_state_history = {}
        HISTORICAL_QUERY_NUM = 0
        query_num = 0
        #history_query_num = 0

        print(f"Initializing chat, start query number: {query_num}")
 
    # The case where USER_QUERY_HISTORY history is enabled but file DNE is handled by testing if the file exists, 
    # The case where USER_QUERY_HISTORY is not enabled is handled by intializing an empty dict and query_num = 0
    # The case where USER_QUERY_HISTORY gives actual queries handles it by upserting history to the history store 

    # Defining the desired prior history state to search over
    DESIRED_HISTORY_WINDOW_SIZE = 3
    # Defining the desired number of context chunks to retrieve for the current query (top_k)
    DESIRED_CONTEXT_CHUNKS_TOP_K = 5
    
    # An example user interaction chain
    
    test_query_preload = [
    'Please explain batch normalization.',                                    # 1
    'How does it relate to layer normalization?',                             # 2
    'Can you explain the advantages of each method?',                         # 3
    'What are some real-world applications of batch normalization?',          # 4
    'How does batch normalization affect the performance of neural networks?',# 5
    'What are the key differences between normalization techniques?',         # 6
    # Hard topic shift begins here:
    'What is reinforcement learning and how does it work?',                   # 7
    'Can you explain the exploration-exploitation tradeoff?',                 # 8
    'What are some popular algorithms used in reinforcement learning?',       # 9
    'asdfadsgfasd',                                                           # 10
    'exit'                                                                    # end
    ]

    rag_scheduler_preload =     [False,  True,  True,  True, False,  True,  True, False,  True, False, False]
    history_scheduler_preload = [ True,  True,  True, False, False,  True, False,  True,  True,  True, False] 
    hybrid_scheduler_preload =  [False, False,  True, False, False,  True, False, False,  True, False, False]
    topic_retrieval_preload =   [False,  True,  True,  True,  True, False, False,  True, False, False, False]
    
    while True:
        # This will be updated by front end before every query
        RAG_SWITCH = True
        RAG_SWITCH = rag_scheduler_preload[query_num - HISTORICAL_QUERY_NUM]
        #RAG_SWITCH = (int(input('1 for RAG, 0 for no RAG')) == 1)
        
        HISTORY_SWITCH = True
        HISTORY_SWITCH = history_scheduler_preload[query_num - HISTORICAL_QUERY_NUM]
        # HISTORY_SWITCH = (int(input('1 for RAG, 0 for no RAG')) == 1)

        BM25_SWITCH = True
        BM25_SWITCH = hybrid_scheduler_preload[query_num - HISTORICAL_QUERY_NUM]

        TOPIC_RETRIEVAL_SWITCH = True
        TOPIC_RETRIEVAL_SWITCH = topic_retrieval_preload[query_num - HISTORICAL_QUERY_NUM]

        # This is a very basic form of history thresholding
        HISTORIC_QUERY_SIMILARITY_THRESHOLD = 0.3 # range [0, 1], setting to 0 is naive window, 1 is basically excluding everything

        # this will save error from occuring if RAG_SWITCH = FALSE and BM25_SWITCH = True
        if not RAG_SWITCH:
            BM25_SWITCH = False

        BM25_RRF_CONSTANT = None if (BM25_SWITCH == False) else 60
        BM25_MULTIPLIER = 10

        # Get user query
        QUERY_TEXT = 'Please explain batch normalization.'
        #QUERY_TEXT = input("\nEnter your question (or type 'exit' to quit): ")
        #QUERY_TEXT = test_query_preload[query_num - HISTORICAL_QUERY_NUM]

        query_start_timestamp = datetime.now().isoformat()
        query_processing_start_time = time.time()

        if QUERY_TEXT.lower() == 'exit':
            break

        # This placement does matter
        query_num += 1

        # Declare a new state in the history state dictionary corresponding to the curent query number
        user_query_state_history[query_num] = {
            # Specific Current Query Details
            "query_number": query_num, # int: The current query number w.r.t user's chat inputs
            "query_text": QUERY_TEXT, # str: The user input query text for the current state
            "query_embedding" : None, # Optional[list[float]]: the query text embedding for semantic search against other queries / context / historical context

            # User Input Values
            "rag_used": RAG_SWITCH, # bool: Indicates whether RAG was used for this query
            "history_used" : HISTORY_SWITCH, # bool: Indicates whether HISTORY was used for this query
            "rag_and_history_used" : RAG_SWITCH and HISTORY_SWITCH, # bool: Indicates if both switches were active for this query
            "bm25_used" : BM25_SWITCH, # bool: Indicates whether bm25 was used for this query
            "topic_retrieval_used" : TOPIC_RETRIEVAL_SWITCH, # bool: Indicates whether topic level retrieval was used for this query
            "desired_lookback_window_size" : DESIRED_HISTORY_WINDOW_SIZE, # int: the desired input lookback window size
            "actual_lookback_window_size" : 0, # int: the size of the actual lookback window after taking into account history length
            "desired_top_k_chunks" : DESIRED_CONTEXT_CHUNKS_TOP_K, # int: the desired number of context chunks to actually examine

            # Query Start and End Time
            "query_start_time" : query_start_timestamp, # str: ISO 8601 timestamp at the start of the query
            "query_finish_time" : None, # Optional[str]: ISO 8601 timestamp at the end of query
            "query_processing_time" : None, # Optional[float]: Duration of query in seconds
            
            # Response Details
            "context_ids_utilized" : None, # Optional[list[int]]: The context ids actually utilized for the response
            "context_ids_source" : None, # Optional[list[int]]: A mask that indicates the source of each of `context_ids_utilized`
            "current_state_hybrid_fused_scores" : None, # Optional[list[float]] # This is a list of fused scores with index corresponding to hybrid ranked retrieved ids
            "response_text": None, # Optional[str]: The model-generated response
            "prompt_token_count" : None, # Optional[int]: Number of tokens in the input prompt
            "candidates_token_count" : None, # Optional[int]: Number of tokens used by LLM for response generation
            "total_token_count" : None, # Optional[int]: Total tokens processed by LLM
            "system_prompt_used": None, # Optional[str]: System Prompt used to generate the current response
            "dynamic_prompt_body": None, # Optional[str]: Generated prompt passed to the LLM based on history/context utilizing `build_historical_query_response_thread` function

            ############## 
            # Requires Retrieval (RAG_SWITCH = True) from here and below
            ##############

            # Retrieval Current Query Context Ids / Scores
            "current_state_context_ids": None, # Optional[list[int]]: Context chunk IDs retrieved for this query
            "current_state_context_scores" : None, # Optional[list[float]]: Similarity scores per chunk
            "retrieval_top_k": None,  # Optional[int]: How many chunks were requested
            "retrieval_method": None,  # Optional[str]: Retrieval method used e.g., "semantic", "bm25+rerank", etc.
            "avg_similarity_to_context": None,  # Optional[float]: Average similarity across retrieved context
            "max_similarity_to_context": None,  # Optional[float]: Max similarity score among retrieved chunks
            "top_context_id": None,  # Optional[int]: ID of the top retrieved chunk
            "top_context_score": None,  # Optional[float]: Similarity score of the top chunk

            # Utilized History if HISTORY_SWITCH = True
            "considered_prior_state_ids" : None, # Optional[list[int]]: All prior query IDs considered for history filtering
            "utilized_prior_state_ids" : None, # Optional[list[int]]: Prior query IDs actually used in prompt (at the moment, query + response pairs from prior queries are utilized in the prompt)

            # Utilized History if (HISTORY_SWITCH = True) & (RAG_SWITCH = True)
            "filter_similarity_score_threshold" : HISTORIC_QUERY_SIMILARITY_THRESHOLD, # Optional[float]: Similarity threshold between current query and prior queries used to filter prior queries
            "filter_similarity_score_mask" : None, # Optional[list[bool]]: Mask over `considered_prior_state_ids` for those that passed threshold
            
            # Utilized BM25 if (BM25 = True)
            "bm25_RRF_constant": BM25_RRF_CONSTANT, # Optional[int]: This is the RRF (Reciprocal Rank Fusion) constant `k`
            "bm25_multiplier": BM25_MULTIPLIER, # Optional[int]: This is the multiplier for the number of chunks to examine when generating top_k chunks (it is necessary)
            "current_state_bm25_context_ids" : None, # Optional[list[int]]: This is a list of retrieved ids utilizing BM25
            "current_state_bm25_context_scores" : None, # Optional[list[float]] # This is a list of context scores with index corresponding to above ids
            
            # Utilized Topic Level Retrieval Information if (TOPIC_RETRIEVAL_SWITCH = True)
            "topic_retrieval_used": TOPIC_RETRIEVAL_SWITCH,  # bool: Indicates whether topic retrieval was used for this query
            "top_topic_chunk_ids": None,  # Optional[list[int]]: The selected top topic chunk IDs retrieved by topic-level search
            "top_topic_chunk_similarity": None,  # Optional[list[float]]: The similarity scores for the selected top topic chunks
            "weight_coherence": None,  # Weight for internal topic coherence (higher is better)
            "weight_query_relevance": None,  # Weight for how well the topic matches the query
            "weight_chunk_count": None,  # Weight for the number of chunks in the topic (higher is better)

            # Experiment Tag
            "experiment_tag" : "user_query_history_logging", # str: Tag to label the experiment type or system mode (e.g., production vs testing)
        }

        # We need to find out how large our actual history window can be, so we have to truncate it if there are not enough prior states to perform a full lookback
        #num_stored_history_count = QDRANT_CLIENT.count(collection_name=HISTORY_COLLECTION).count

        # Get prior query numbers in descending order, excluding current one
        prior_query_numbers = sorted(
            [k for k in user_query_state_history.keys() if k < query_num],
            reverse=True
        )

        # Limit to desired window size
        #actual_history_id_window = prior_query_numbers[:DESIRED_HISTORY_WINDOW_SIZE]
        #actual_history_id_window_length = len(actual_history_id_window)

        actual_history_id_window = sorted(list(range(max(1, query_num - DESIRED_HISTORY_WINDOW_SIZE), query_num, 1)), reverse=True)
        actual_history_id_window_length = len(actual_history_id_window)

        #print(f'{dict((k, v) for k, v in new_user_history[query_num].items() if k != "query_embedding")}')
        print('\n###########################################')
        print(f'############### FOR QUERY {query_num} ###############')
        print('###########################################')
        print(f"#### Rag Switch: {RAG_SWITCH}")
        print(f"#### History Switch: {HISTORY_SWITCH}")
        print(f"#### BM25 Switch: {BM25_SWITCH}")
        print(f'####\n############### QUERY: \n#### {QUERY_TEXT}\n####')

        system_prompt = ''

        # If user does not enable `RAG_SWITCH`, just get a generic response from the LLM
        if not RAG_SWITCH:
            # If user does not enable `HISTORY_SWITCH`, treat the prompt as standalone (only pass `QUERY_TEXT`)
            if not HISTORY_SWITCH:

                #system_prompt = """Please provide a comprehensive, well reasoned response to the query.
                #                Draw from innate knowledge to synthesize a coherent response.
                #                If there are metrics or equations that are relevant to the query, include them to your response.\n"""

                # Generate a Response with no lookback
                dynamic_prompt_body = build_historical_query_response_thread(user_history=user_query_state_history, 
                                                                        current_query_num=query_num, 
                                                                        window_size=0,
                                                                        history_switch=HISTORY_SWITCH,
                                                                        naive_window=True,)
                                                                        #top_k_chunks_per_prior_query=2)

            # If user does enable `HISTORY_SWITCH`, perform a naive lookback across the history window
            elif HISTORY_SWITCH:

                #system_prompt = """Please provide a comprehensive, well reasoned response to the query using the provided context. 
                #                Draw from the history and innate knowledge to synthesize a coherent result.
                #                If there are metrics or equations that are relevant to the query, include them to your response.\n"""
                
                # Generate a Response with lookback
                dynamic_prompt_body = build_historical_query_response_thread(user_history=user_query_state_history, 
                                                                        current_query_num=query_num, 
                                                                        window_size=actual_history_id_window_length,
                                                                        history_switch=HISTORY_SWITCH,
                                                                        naive_window=True,)
                                                                        #top_k_chunks_per_prior_query=2)
                
                # Our Naive Approach considers our actual history window, then simply passes through all of it as context
                user_query_state_history[query_num].update({
                    "considered_prior_state_ids" : actual_history_id_window,
                    "utilized_prior_state_ids" : actual_history_id_window,
                    "actual_lookback_Window_size" : actual_history_id_window_length, # the size of the actual lookback window after taking into account history length
                }
                )

        # If user does enable `RAG_SWITCH`, retrieve relevant context from the LLM
        elif RAG_SWITCH:

            top_k_chunks_search_num = DESIRED_CONTEXT_CHUNKS_TOP_K

            if BM25_SWITCH:

                top_k_chunks_search_num *= BM25_MULTIPLIER

                def get_bm25_top_k(query, top_k):
                    # Example query
                    tokenized_query = nltk.word_tokenize(query.lower())  # Tokenize the query

                    # Get BM25 scores for each chunk in the corpus
                    chunk_scores = BM25_SEARCH_FUNCTION.get_scores(tokenized_query)

                    # Get the top k document indices based on sorted BM25 scores
                    top_k_indices = sorted(range(len(chunk_scores)), key=lambda i: chunk_scores[i], reverse=True)[:top_k]

                    top_k_score_list = list(chunk_scores[top_k_indices])

                    top_k_score_list = [float(score) for score in top_k_score_list]

                    return list(top_k_indices), top_k_score_list
                
                bm25_chunk_ids, bm25_chunk_scores = get_bm25_top_k(query=QUERY_TEXT, top_k=top_k_chunks_search_num)

                user_query_state_history[query_num].update({
                    "bm25_RRF_constant": BM25_RRF_CONSTANT,
                    "current_state_bm25_context_ids": bm25_chunk_ids, # BM25 `DESIRED_CONTEXT_CHUNKS_TOP_K` chunk ids
                    "current_state_bm25_context_scores": bm25_chunk_scores # BM25 `DESIRED_CONTEXT_CHUNKS_TOP_K` chunk scores
                }
                )

            #rag_timestamp = datetime.now().isoformat()

            # Get the query text embeddings for the current state
            user_text_query_embedding = get_query_embedding(model=EMBEDDING_MODEL,
                                                tokenizer=TOKENIZER,
                                                query_text=QUERY_TEXT,
                                                device=DEVICE
            )

            # Get the context chunks for the current state
            current_query_context_chunks = get_context_chunks(qdrant_client=QDRANT_CLIENT, 
                                                        norm_query_embedding=user_text_query_embedding, 
                                                        num_chunks=top_k_chunks_search_num,
                                                        my_collection=CHUNK_COLLECTION
            )

            # Get the context chunk ids for the current state
            current_state_context_chunk_ids = []
            for i, chunk in enumerate(current_query_context_chunks):
                current_state_context_chunk_ids.append(chunk.id)  # Collect the ID

            # Compute the current state's average context score (sort of a general measurement, can be refined later)
            current_state_context_scores = [chunk.score for chunk in current_query_context_chunks]

            # Declare a new state in the history state dictionary corresponding to the curent query number
            user_query_state_history[query_num].update({
                    "query_embedding" : user_text_query_embedding, # the query text embedding for semantic search against other queries / context / historical context
                    "current_state_context_ids": current_state_context_chunk_ids, # the context ids of the current state's context 
                    "current_state_context_scores" : current_state_context_scores, # the mean similarity score of the current state context cosine similarities
            }
            )

            # Compute additional retrieval metadata
            user_query_state_history[query_num].update({
                "retrieval_top_k": len(current_state_context_chunk_ids),  # usually 5
                "retrieval_method": "semantic",  # change if you use a different method later
                "avg_similarity_to_context": sum(current_state_context_scores) / len(current_state_context_scores) if current_state_context_scores else 0.0,
                "max_similarity_to_context": max(current_state_context_scores) if current_state_context_scores else 0.0,
                "top_context_id": current_state_context_chunk_ids[0] if current_state_context_chunk_ids else None,
                "top_context_score": current_state_context_scores[0] if current_state_context_scores else None,
            }
            )

            #print('####', bm25_chunk_ids)
            #print('####', bm25_chunk_scores)

            #return bm25_chunk_ids, bm25_chunk_scores

            # If the user has not enabled `HISTORY_SWITCH`, or the query is the first query (as in, there is no existing history to draw from)
            if not HISTORY_SWITCH or query_num == 1:
                
                #system_prompt = """Please provide a comprehensive, well reasoned response to the query using the provided context. 
                #                Draw from the context and innate knowledge to synthesize a coherent result. You need to both summarize and expand upon
                #                the current context in order to relate it to the user's current query.
                #                If there are metrics or equations that are relevant to the query, include them to your response.\n"""

                # Generate a response with context, but no lookback
                #initial_prompt = build_historical_query_response_thread(user_history=user_query_state_history, current_query_num=query_num, window_size=0, naive_window=True)

                # Generate a Response with no lookback
                dynamic_prompt_body = build_historical_query_response_thread(user_history=user_query_state_history, 
                                                                        current_query_num=query_num, 
                                                                        window_size=0,
                                                                        history_switch=HISTORY_SWITCH,
                                                                        naive_window=False
                )
            
            # If the user has enabled `HISTORY_SWITCH` and there is history to draw from
            elif HISTORY_SWITCH:
                #system_prompt = """Please provide a comprehensive, well reasoned response to the query using the provided context. 
                #                Draw from the context and innate knowledge to synthesize a coherent result. You need to both summarize and expand upon
                #                the provided historical query/response pairs and current context in order to relate it to the user's current query.
                #                If there are metrics or equations that are relevant to the query, include them to your response.\n"""

                scored_prior_state_results = current_query_to_prior_queries_filter(qdrant_client=QDRANT_CLIENT,
                                                                                norm_query_embedding=user_text_query_embedding,
                                                                                num_chunks=DESIRED_HISTORY_WINDOW_SIZE,
                                                                                my_collection=HISTORY_COLLECTION,
                                                                                filter_ids_on=actual_history_id_window
                )

                # Step 2: Build ID-to-score map from returned results
                scored_result_map = {int(hit.id): hit.score for hit in scored_prior_state_results}

                #"filter_similarity_score_mask" : HISTORIC_QUERY_SIMILARITY_THRESHOLD,

                # Step 3: Build mask over `actual_history_id_window`
                similarity_score_mask = [
                    scored_result_map.get(query_id, 0.0) >= HISTORIC_QUERY_SIMILARITY_THRESHOLD # this has range [0, 1]
                    for query_id in actual_history_id_window
                ]

                # Step 4: Extract final prior query IDs that passed threshold
                utilized_prior_query_ids = [
                    query_id for query_id, keep in zip(actual_history_id_window, similarity_score_mask) if keep
                ]

                # Use the boolean mask to get the actual query IDs
                relevant_prior_query_ids = [query_id for query_id, passed in zip(actual_history_id_window, utilized_prior_query_ids) if passed]

                user_query_state_history[query_num].update({
                    "filter_similarity_score_mask" : similarity_score_mask,
                    "considered_prior_state_ids" : actual_history_id_window,
                    "utilized_prior_state_ids" : utilized_prior_query_ids, # the context ids of the semantically searched prior states context that was utilized as additional historical context in the current state's full prompt
                    "actual_lookback_window_size" : actual_history_id_window_length, # the size of the actual lookback window after taking into account history length

                }
                )

                # Generate a Response with RAG and lookback (difference here is that user_query_state_history is a filtered)
                dynamic_prompt_body = build_historical_query_response_thread(user_history=user_query_state_history, 
                                                                        current_query_num=query_num, 
                                                                        window_size=actual_history_id_window_length,
                                                                        history_switch=HISTORY_SWITCH,
                                                                        naive_window=False,
                                                                        relevant_prior_query_ids=relevant_prior_query_ids)
                                                                        #top_k_chunks_per_prior_query=2)

        user_query_state_history[query_num].update({
            "system_prompt_used": system_prompt,
            "dynamic_prompt_body": dynamic_prompt_body
        }
        )

        def context_thread_builder(context_ids=[], hybrid=False, topic_retrieval=False):
            current_query_context_prompt = []

            # Adjust the prompt based on the flags for `hybrid` and `topic_retrieval`
            if topic_retrieval:
                #if hybrid:
                    #current_query_context_prompt.append(f"\nContext Chunk # indicates hybrid RRF (semantic + bm25) ranking of context chunks in descending order for the top topic:")
                #else:
                current_query_context_prompt.append(f"\nContext Chunk # indicates ranking of context chunks in descending order for the top topic:")
            else:
                #if hybrid:
                #    current_query_context_prompt.append(f"\nContext Chunk # indicates hybrid RRF (semantic + bm25) ranking of context chunks in descending order:")
                #else:
                current_query_context_prompt.append(f"\nContext Chunk # indicates semantic ranking of context chunks in descending order.")

            #if hybrid:
            #    current_query_context_prompt.append(f"\nContext Chunk # indicates hybrid RRF (semantic + bm25) ranking of context chunks in descending order:")
            #else:
            #    current_query_context_prompt.append(f"\nContext Chunk # indicates semantic ranking of context chunks in descending order.")

            # need to scroll QDRANT_CLIENT to get the text of the payloads associated with each context_id
            context_payloads = QDRANT_CLIENT.retrieve(collection_name=CHUNK_COLLECTION, ids=context_ids)

            context_text_list = [chunk.payload['text'] for chunk in context_payloads]

            for i, context_text in enumerate(context_text_list):
                current_query_context_prompt.append(f"\nCONTEXT CHUNK {i}: {context_text}")

            return "\n".join(current_query_context_prompt)

        if RAG_SWITCH:
            # If we have Hybrid Search, we have to handle it by choosing from both our sources for choosing context.
            if BM25_SWITCH:

                context_ids_from_semantic_sim = user_query_state_history[query_num]['current_state_context_ids']
                context_scores_from_semantic_sim = user_query_state_history[query_num]['current_state_context_scores']

                context_ids_from_bm25 = user_query_state_history[query_num]['current_state_bm25_context_ids']
                context_scores_from_bm25 = user_query_state_history[query_num]['current_state_bm25_context_scores']
                
                # Step 1: Initialize a dictionary to store the fused RRF scores and sources
                fused_scores = {}
                source_trace = {}  # This will track whether a document comes from 'semantic', 'bm25', or 'both'

                # Step 2: Apply Reciprocal Rank Fusion (RRF) to combine the rank positions
                # For each rank in both semantic and BM25 lists, apply the reciprocal rank formula

                # Apply RRF to semantic ranks (top_k elements)
                for rank, doc_id in enumerate(context_ids_from_semantic_sim, start=1):
                    # Add the reciprocal rank score for semantic
                    fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (BM25_RRF_CONSTANT + rank)
                    # Track the source as 'semantic'
                    source_trace[doc_id] = source_trace.get(doc_id, 'semantic')

                # Apply RRF to BM25 ranks (top_k elements)
                for rank, doc_id in enumerate(context_ids_from_bm25, start=1):
                    # Add the reciprocal rank score for BM25
                    fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (BM25_RRF_CONSTANT + rank)
                    # Track the source as 'bm25', or update it to 'both' if it exists in both
                    if doc_id in source_trace:
                        source_trace[doc_id] = 'both'  # If it exists in both, mark it as 'both'
                    else:
                        source_trace[doc_id] = 'bm25'  # Otherwise, mark it as 'bm25'

                # Step 3: Sort the documents based on their fused scores
                top_k_indices = sorted(fused_scores, key=fused_scores.get, reverse=True)[:DESIRED_CONTEXT_CHUNKS_TOP_K]

                # Step 4: Generate the fused top_k list with source labels and fused scores
                fused_top_k_list = []
                for idx in top_k_indices:
                    fused_top_k_list.append({
                        'context_id': idx,
                        'fused_score': fused_scores[idx],
                        'source': source_trace[idx]
                    }
                    )

                # Step 5: Print the fused top_k list
                for item in fused_top_k_list:
                    print(f"Context ID: {item['context_id']}, Fused RRF Score: {item['fused_score']:.3f}, Source: {item['source']}")

                # Extracting the context_ids_utilized, context_ids_source, context_ids_scores from the fused_top_k_list
                context_ids_utilized = [item['context_id'] for item in fused_top_k_list]
                context_ids_source = [item['source'] for item in fused_top_k_list]
                context_ids_fused_scores = [item['fused_score'] for item in fused_top_k_list]

                # Updating utlized_ids
                user_query_state_history[query_num].update({
                    "context_ids_utilized" : context_ids_utilized, # Optional[list[int]]: The context ids actually utilized for the response
                    "context_ids_source" : context_ids_source, # Optional[list[int]]: A mask that indicates the source of each of `context_ids_utilized`
                    "current_state_hybrid_fused_scores" : context_ids_fused_scores, # Optional[list[float]] # This is a list of fused scores with index corresponding to hybrid ranked retrieved ids
                }
                )

            else:
                input_current_state_context_ids = user_query_state_history[query_num]['current_state_context_ids']

                user_query_state_history[query_num].update({
                    "context_ids_utilized" : input_current_state_context_ids, # Optional[list[int]]: The context ids actually utilized for the response
                    "context_ids_source" : ['semantic'] * len(input_current_state_context_ids) # Optional[list[str]]: A mask that indicates the source of each of `context_ids_utilized`
                }
                )
            
            context_ids_for_prompt = user_query_state_history[query_num]['context_ids_utilized']

            current_context_prompt_body = context_thread_builder(context_ids=context_ids_for_prompt, hybrid=BM25_SWITCH)

        else:
            current_context_prompt_body = ''

        if RAG_SWITCH and TOPIC_RETRIEVAL_SWITCH:
            chunk_to_topic_chunks_mapper, chunk_to_topic_info_mapper = topic_level_context_retrieval(current_query_context_chunks=current_query_context_chunks,
                                            QDRANT_CLIENT=QDRANT_CLIENT,
                                            CHUNK_COLLECTION=CHUNK_COLLECTION)
            
            print('\nMy Single Chunk to Topic Mapper:', chunk_to_topic_chunks_mapper)

            # Initialize an empty dictionary to store the deduplicated values
            deduplicated_chunk_to_topic_chunks_mapper = {}
            deduplicated_chunk_to_topic_info_mapper = {}

            # Initialize a temporary mapper that allows for the count of how many times a topic is mentioned
            temp_mapper = {}

            # Initialize a set to track already encountered keys
            encountered_keys = []

            # Iterate through the original dictionary to deduplicate it
            for key, values in chunk_to_topic_chunks_mapper.items():
                if key not in encountered_keys:
                    deduplicated_chunk_to_topic_chunks_mapper[key] = values
                    deduplicated_chunk_to_topic_info_mapper[key] = chunk_to_topic_info_mapper[key]

                    # Initialize the topic presence count
                    deduplicated_chunk_to_topic_info_mapper[key].update({'topic_presence': 1})

                    for value in values:
                        temp_mapper[value] = key

                    encountered_keys.extend(values)
                
                else:
                    # If the chunk is encountered again, increment the topic presence for the associated topic
                    topic_group_original_chunk_id = temp_mapper[key]
                    deduplicated_chunk_to_topic_info_mapper[topic_group_original_chunk_id]['topic_presence'] += 1

            #print('\nMy Deduplicated Single Chunk to Topic Mapper:', deduplicated_chunk_to_topic_chunks_mapper)
            #print('\nMy Single Chunk to Arxiv ID Mapper:', deduplicated_chunk_to_topic_info_mapper)

            # Rank topics based on topic_presence (higher count means higher priority)
            ranked_topics = sorted(deduplicated_chunk_to_topic_info_mapper.items(),
                                key=lambda x: x[1]['topic_presence'], reverse=True)

            # Display the top-ranked topics
            #print("\nRanked Topics based on Topic Presence:")
            #for topic_id, topic_info in ranked_topics:
                #print(f"Topic: {topic_info['topic']}, Topic Presence: {topic_info['topic_presence']}, Arxiv ID: {topic_info['arxiv_id']}")

            # Function to retrieve vectors from Qdrant by their IDs
            def get_vectors_from_qdrant_by_ids(ids):
                # Retrieve points (vectors) from Qdrant by IDs
                response = QDRANT_CLIENT.retrieve(
                    collection_name=CHUNK_COLLECTION,
                    ids=ids,  # List of IDs of the vectors you want to retrieve
                    with_vectors=True
                )
                
                # Convert the Qdrant response into a dictionary of {chunk_id: vector}
                chunk_vectors = {point.id: point.vector for point in response}
                
                return chunk_vectors

            from sklearn.metrics.pairwise import cosine_similarity

            # Function to calculate average similarity for a group of chunks (a topic)
            def calculate_average_similarity_for_topic(topic_chunk_ids, chunk_embeddings):
                # Extract embeddings for the chunks in the given topic
                topic_chunk_embeddings = [chunk_embeddings[chunk_id] for chunk_id in topic_chunk_ids]
                
                # Number of chunks (vectors) in the topic
                n = len(topic_chunk_ids)
                
                # If there is only one chunk, no similarity calculation is needed (returning 0.8 to prevent unrealistically large values)
                if n <= 1:
                    return 0.5  # or return 0, depending on how you want to handle single chunk topics
                
                # Calculate pairwise cosine similarities
                similarity_matrix = cosine_similarity(topic_chunk_embeddings)
                
                # Flatten the similarity matrix (convert to 1D)
                flattened_similarity = similarity_matrix.flatten()

                # The diagonal elements correspond to self-similarity and should be removed
                diagonal_indices = [i * (n + 1) for i in range(n)]  # Indices of the diagonal elements (i * n + i)
                
                # Remove diagonal elements from the flattened similarity array
                flattened_similarity = np.delete(flattened_similarity, diagonal_indices)

                # Compute the average cosine similarity (mean of non-diagonal elements)
                average_similarity = np.mean(flattened_similarity)
                
                return average_similarity
            
            # Function to calculate the average similarity between the query embedding and all chunk embeddings
            def calculate_average_query_similarity(query_embedding, chunk_id_vector_pairs):
                # List to store cosine similarity between the query and each chunk
                similarity_scores = []

                # Iterate through each chunk ID and vector in the chunk_id_vector_pairs dictionary
                for chunk_id, chunk_vector in chunk_id_vector_pairs.items():
                    # Compute cosine similarity between the query embedding and the chunk vector
                    similarity = cosine_similarity([query_embedding], [chunk_vector])[0][0]
                    
                    # Append the similarity score to the list
                    similarity_scores.append(similarity)
                
                # Calculate the average similarity
                average_similarity = np.mean(similarity_scores)
                
                return average_similarity, similarity_scores

            for key, values in deduplicated_chunk_to_topic_chunks_mapper.items():
                #print(f"Processing Topic ID: {key}, Chunk IDs: {values}")

                my_chunk_id_vector_pairs = get_vectors_from_qdrant_by_ids(values)

                #print(my_chunk_id_vector_pairs[values[0]])
                
                inter_pool_average_chunk_similarity = calculate_average_similarity_for_topic(values, my_chunk_id_vector_pairs)
                
                query_to_chunk_pool_average_similarity, query_to_chunk_similarity_scores = calculate_average_query_similarity(
                    user_text_query_embedding, my_chunk_id_vector_pairs
                )
                
                # Number of chunks in the topic (size of 'values')
                number_of_chunks = len(values)

                # Output for debugging
                #print('\nAverage Inter-Chunk Pool Similarity: ', inter_pool_average_chunk_similarity)
                #print('Average Query-to-Chunk Pool Similarity: ', query_to_chunk_pool_average_similarity)
                #print('Number of Chunks in this Topic: ', number_of_chunks)

                # Update the deduplicated_chunk_to_topic_info_mapper with the new fields
                deduplicated_chunk_to_topic_info_mapper[key].update({
                    'inter_pool_average_chunk_similarity': inter_pool_average_chunk_similarity,
                    'query_to_chunk_pool_average_similarity': query_to_chunk_pool_average_similarity,
                    'chunk_pool' : values,
                    'query_to_chunk_similarity_scores': query_to_chunk_similarity_scores,
                    'number_of_chunks': number_of_chunks
                })

            # Print the updated info mapper for verification
            #print("\nUpdated Deduplicated Chunk to Topic Info Mapper:")
            #for topic_id, topic_info in deduplicated_chunk_to_topic_info_mapper.items():
                #print(f"Topic ID: {topic_id}, Info: {topic_info}")

            ############### Finally, calculate the score:

            # Example weights (adjust as necessary)
            weight_coherence = 1 / 3  # Importance of internal topic coherence (higher is better)
            weight_query_relevance = 1 / 3  # Importance of how well the topic matches the query
            weight_chunk_count = 1 / 3  # Importance of the number of chunks in the topic (higher is better)

            # Function to calculate combined score for each topic
            def calculate_combined_score(topic_info):
                # Find the max values for coherence, query relevance, and chunk count across all topics
                max_coherence = max([topic_info['inter_pool_average_chunk_similarity'] for topic_info in deduplicated_chunk_to_topic_info_mapper.values()])
                max_query_relevance = max([topic_info['query_to_chunk_pool_average_similarity'] for topic_info in deduplicated_chunk_to_topic_info_mapper.values()])
                max_chunk_count = max([topic_info['number_of_chunks'] for topic_info in deduplicated_chunk_to_topic_info_mapper.values()])

                # Normalize factors (if necessary)
                normalized_coherence = topic_info['inter_pool_average_chunk_similarity'] / max_coherence
                normalized_query_relevance = topic_info['query_to_chunk_pool_average_similarity'] / max_query_relevance
                normalized_chunk_count = topic_info['number_of_chunks'] / max_chunk_count

                # Calculate combined score (weighted sum of normalized factors)
                combined_score = (weight_coherence * normalized_coherence + 
                                weight_query_relevance * normalized_query_relevance + 
                                weight_chunk_count * normalized_chunk_count)

                return combined_score

            # Process each topic and calculate its combined score
            for key, topic_info in deduplicated_chunk_to_topic_info_mapper.items():
                combined_score = calculate_combined_score(topic_info)
                
                # Store the combined score in the topic info (you can use this score later to retrieve the most relevant topic)
                deduplicated_chunk_to_topic_info_mapper[key].update({'combined_score': combined_score})

            # Sort topics by combined score to retrieve the most relevant one
            sorted_topics = sorted(deduplicated_chunk_to_topic_info_mapper.items(), key=lambda x: x[1]['combined_score'], reverse=True)

            # Output the sorted topics by combined score
            print("\nSorted Topics by Combined Score:")
            for topic_id, topic_info in sorted_topics:
                print(f"Topic ID: {topic_id}, Combined Score: {topic_info['combined_score']}, Topic: {topic_info['topic']}")


            # Function to get top-k chunks based on query similarity scores
            def get_top_k_chunks_for_topic(topic_info, k):
                # Get the similarity scores and chunk IDs from the topic info
                similarity_scores = topic_info['query_to_chunk_similarity_scores']
                chunk_pool = topic_info['chunk_pool']
                
                # Pair chunk IDs with their similarity scores
                chunks_with_similarity = [(chunk_pool[i], similarity_scores[i]) for i in range(len(similarity_scores))]
                
                # Sort the chunks by similarity score in descending order
                sorted_chunks = sorted(chunks_with_similarity, key=lambda x: x[1], reverse=True)
                
                # Select the top-k chunks based on similarity
                top_k_chunks = sorted_chunks[:k]
                
                # Print the top-k chunks for this topic
                print(f"Top {k} Chunks for Topic {topic_info['topic']}:")
                for chunk_id, similarity in top_k_chunks:
                    print(f"Chunk ID: {chunk_id}, Similarity: {similarity}")
                
                return top_k_chunks

            # Access the top topic directly from sorted_topics
            top_topic_info = sorted_topics[0][1]  # Get the topic data from the first sorted topic

            # Get the top-k chunks for the selected topic
            top_k_topic_chunks = get_top_k_chunks_for_topic(top_topic_info, DESIRED_CONTEXT_CHUNKS_TOP_K)

            # Extract chunk IDs and similarity scores
            top_k_chunks_ids = [chunk_id for chunk_id, _ in top_k_topic_chunks]
            top_k_chunks_similarity = [similarity for _, similarity in top_k_topic_chunks]

            # Now, update the topic info with the new fields
            top_topic_info.update({
                'top_k_chunks': top_k_chunks_ids,
                'top_k_chunks_similarity': top_k_chunks_similarity
            }
            )

            # Print the final top topic info with top-k chunks (not the entire `deduplicated_chunk_to_topic_info_mapper`)
            print("\nFinal Top Topic Information:")
            print(top_topic_info)
            print("\nFinal Top Topic Chunk IDs:")
            print(top_topic_info['top_k_chunks'])
            print(top_topic_info['top_k_chunks_similarity'])

            # Determine the source labels based on retrieval method
            if BM25_SWITCH:
                context_ids_source = ['hybrid_topic'] * len(top_k_chunks_ids)  # Hybrid RAG (topic + semantic)
            else:
                context_ids_source = ['semantic_topic'] * len(top_k_chunks_ids)  # Semantic only if RAG and BM25 are used
                
            # Update the log with the topic retrieval details
            user_query_state_history[query_num].update({
                "top_topic_chunk_ids": top_k_chunks_ids,  # List of topic chunk IDs
                "top_topic_chunk_similarity": top_k_chunks_similarity,  # List of similarity scores for the top topic chunks

                # Topic-specific parameters (weights)
                "weight_coherence": weight_coherence,
                "weight_query_relevance": weight_query_relevance,
                "weight_chunk_count": weight_chunk_count,
                
                # Update context-related fields with topic-specific chunk information
                "context_ids_utilized": top_k_chunks_ids,  # Set context_ids_utilized to the topic chunk IDs
                "context_ids_source": context_ids_source,  # Set context_ids_source to the similarity scores
            }
            )

            current_context_prompt_body = context_thread_builder(context_ids=context_ids_for_prompt, hybrid=BM25_SWITCH, topic_retrieval=TOPIC_RETRIEVAL_SWITCH)

        #display_summary(current_user_query_state_history=user_query_state_history[query_num]) 

        #print(current_context_prompt_body)

        response = standalone_prompt_model(llm_model=LLM_MODEL, system_prompt=system_prompt, historical_prompt=dynamic_prompt_body, context_prompt=current_context_prompt_body)
        #response = 'Placeholder Text'

        end_time = time.time()

        query_stop_timestamp = datetime.now().isoformat()
        query_processing_end_time = end_time - query_processing_start_time

        user_query_state_history[query_num].update({
                "response_text" : response.text, # the response text of the model
                "prompt_token_count" : response.usage_metadata.prompt_token_count, # token count for input prompt to LLM
                "candidates_token_count" : response.usage_metadata.candidates_token_count, # utilized tokens from input prompt to LLM to generate response
                "total_token_count" : response.usage_metadata.total_token_count, # total tokens processed by LLM
                "query_finish_time" : query_stop_timestamp,
                "query_processing_time" : query_processing_end_time,
        }
        )
        
        response_plaintext = markdown_to_text(response.text)
        response_plaintext = response_plaintext.split('\n')
        for line in response_plaintext: 
            print('#### ', line)

        print('###########################################')

        ####################### GET THE RESPONSE FROM HERE ###############################
        # Response is `user_query_state_history["response_text"]`

        if RAG_SWITCH:
            upsert_history_to_vector_store(qdrant_client=QDRANT_CLIENT, 
                        collection_name=HISTORY_COLLECTION, 
                        history=user_query_state_history[query_num],
                        batch=False # upsert current query state
            )

        save_chat_json(user_query_state_history[query_num])

    save_chat_pkl_by_embedding(user_query_state_history, 
                                embedded_path='user_output/user_embedded_history.pkl',
                                non_embedded_path='user_output/user_non_embedded_history.pkl'
    )

if __name__ == '__main__':
    main()