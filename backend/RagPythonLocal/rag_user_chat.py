# %% [markdown]
# # Import Dependencies
# 

# %%
# Import necessary functions from the uploaded files
import os
from final_response_front_end_main import initialize_system, process_query
from final_response_front_end_main import load_query_history
from user_history_utils import save_chat_pkl_by_embedding
from user_history_utils import save_chat_json

# %% [markdown]
# # Setups:
# 
# ### Below are the various setups of environment variables and functions that are required for **EVERY SESSION**.

# %% [markdown]
# ## Intialize System Environment (do this once PER BOOT OF SYSTEM)

# %%
# Initialize the environment
DEVICE, TOKENIZER, EMBEDDING_MODEL, LLM_MODEL, LLM_SYSTEM_PROMPT, QDRANT_CLIENT, CHUNK_COLLECTION, HISTORY_COLLECTION, BM25_SEARCH_FUNCTION, _, _, _ = initialize_system()

# %% [markdown]
# ## Initialize a user's chat (do this ONCE PER SESSION / when the next user job is DIFFERENT from the prior user's job)

# %%
def initialize_user_chat(
                        task_folder='saved_chats', # The top level folder, where user chats are located
                        user_output_folder='test_user', # The next level folder that stores all of a user's chats (define user names here)
                        saved_chats_topic_name='ey_lmao1', # The bottom level folder that stores a specific chat from the user (contains .json / .pkl files) 
    ):

    directory = f'user_output/{task_folder}/{user_output_folder}/{saved_chats_topic_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define the name of the chat, and the .json / .pkl files that will be saved with it
    chat_json_name = f"{saved_chats_topic_name}.json"
    chat_embedded_history_name = 'user_embedded_history.pkl'
    chat_non_embedded_history_name = 'user_non_embedded_history.pkl'

    # Define the path of the json / pkl files
    final_json_path = os.path.join(directory, chat_json_name)
    final_embedded_history_path = os.path.join(directory, chat_embedded_history_name)
    final_non_embedded_history_path = os.path.join(directory, chat_non_embedded_history_name)

    # Load the chat history given these paths
    user_query_state_history, query_num, HISTORICAL_QUERY_NUM = load_query_history(QDRANT_CLIENT=QDRANT_CLIENT, 
            HISTORY_COLLECTION=HISTORY_COLLECTION,
            chat_embedded_history_path=final_embedded_history_path,
            chat_non_embedded_history_path=final_non_embedded_history_path
            )

    initialized_chat_settings = {
        'user_chat_json_path': final_json_path,
        'user_embedded_history_path': final_embedded_history_path,
        'user_non_embedded_history_path': final_non_embedded_history_path,
        'query_num': query_num,
        'historical_query_num': HISTORICAL_QUERY_NUM,
        'user_query_state_history': user_query_state_history
    }

    return initialized_chat_settings


# %% [markdown]
# ## Call Response Generation Function (DO THIS AS MANY TIMES AS YOU WANT PER SESSION)

# %%
def call_back_end_response_generation(
                                      user_query_state_history: dict,
                                      desired_history_window_size: int,
                                      desired_context_chunks_top_k: int,
                                      rag_switch: bool,
                                      history_switch: bool,
                                      bm25_switch: bool,
                                      topic_retrieval_switch: bool,
                                      historic_query_similarity_threshold: float, # [0, 1] range (filter)
                                      query_text: str,
                                      query_num: int,
    ):

    # Call the `process_query()` function with the inputs
    user_query_state_history[query_num] = process_query(
        desired_history_window_size, 
        desired_context_chunks_top_k, 
        rag_switch, 
        history_switch, 
        bm25_switch, 
        topic_retrieval_switch, 
        historic_query_similarity_threshold, 
        query_text, 
        user_query_state_history,
        query_num, 
        QDRANT_CLIENT, 
        CHUNK_COLLECTION,
        HISTORY_COLLECTION,
        LLM_MODEL,
        LLM_SYSTEM_PROMPT,
        DEVICE,
        EMBEDDING_MODEL, 
        TOKENIZER,
        BM25_SEARCH_FUNCTION
    )

    return user_query_state_history[query_num]

# %% [markdown]
# ## Input a Query to the Backend (DO THIS AS MANY TIMES AS YOU WANT PER SESSION)

# %%
def input_query(
                user_chat_settings: dict={}, # the initial settings used to route / initialize the chat  
                frontend_inputs: dict={},
                end_session: bool=False # the user inputs for a given query                  
    ):

    MY_USER_JSON_PATH = user_chat_settings['user_chat_json_path']

    query_num = user_chat_settings['query_num']
    #HISTORICAL_QUERY_NUM = user_chat_settings['historical_query_num']
    user_query_state_history = user_chat_settings['user_query_state_history']

    USER_INPUT_QUERY = frontend_inputs['USER_INPUT_QUERY']
    DESIRED_HISTORY_WINDOW_SIZE = frontend_inputs['DESIRED_HISTORY_WINDOW_SIZE']
    DESIRED_CONTEXT_CHUNKS_TOP_K = frontend_inputs['DESIRED_CONTEXT_CHUNKS_TOP_K']
    RAG_SWITCH = frontend_inputs['RAG_SWITCH']
    HISTORY_SWITCH = frontend_inputs['HISTORY_SWITCH']
    BM25_SWITCH = frontend_inputs['BM25_SWITCH']
    TOPIC_RETRIEVAL_SWITCH = frontend_inputs['TOPIC_RETRIEVAL_SWITCH']
    HISTORIC_QUERY_SIMILARITY_THRESHOLD = frontend_inputs['HISTORIC_QUERY_SIMILARITY_THRESHOLD']

    # Increment the query by 1
    query_num += 1

    user_query_state_history[query_num] = call_back_end_response_generation(
                        user_query_state_history=user_query_state_history,
                        desired_history_window_size=DESIRED_HISTORY_WINDOW_SIZE,
                        desired_context_chunks_top_k=DESIRED_CONTEXT_CHUNKS_TOP_K,
                        rag_switch=RAG_SWITCH,
                        history_switch=HISTORY_SWITCH,
                        bm25_switch=BM25_SWITCH,
                        topic_retrieval_switch=TOPIC_RETRIEVAL_SWITCH,
                        historic_query_similarity_threshold=HISTORIC_QUERY_SIMILARITY_THRESHOLD,
                        query_text=USER_INPUT_QUERY,
                        query_num=query_num
    )

    user_chat_settings['user_query_state_history'][query_num] = user_query_state_history[query_num]
    user_chat_settings['query_num'] = query_num

    save_chat_json(user_query_state_history[query_num], file_path=MY_USER_JSON_PATH)
    
    print(f"User State Length: {len(user_query_state_history)}")
    print(f"\nResponse: {user_query_state_history[query_num]['response_text']}\n")

    return user_chat_settings

# %% [markdown]
# ## Save the Chat Session / Move to next user's job

# %%
def end_user_session(user_chat_settings):

    # Save the PKL file at the end of the session
    MY_USER_EMBEDDED_HISTORY_PATH = user_chat_settings['user_embedded_history_path']
    MY_USER_NON_EMBEDDED_HISTORY_PATH = user_chat_settings['user_non_embedded_history_path']
    USER_QUERY_STATE_HISTORY = user_chat_settings['user_query_state_history']

    save_chat_pkl_by_embedding(
        user_query_state_history=USER_QUERY_STATE_HISTORY,
        embedded_path=MY_USER_EMBEDDED_HISTORY_PATH,
        non_embedded_path=MY_USER_NON_EMBEDDED_HISTORY_PATH
    )
