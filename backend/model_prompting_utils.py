def build_historical_query_response_thread(user_history={}, current_query_num=1, window_size=0, history_switch=False, naive_window=True, relevant_prior_query_ids=[]):
    history_context = []
    history_used_in_prompt_bool = False

    # if the user has decided to enable history
    if history_switch:

        # and the user has selected a lookback window size greater than 0
        if window_size > 0:

            ############## VERY IMPORTANT ##############
            # I am going to add a condition here that controls fundamentally how the model retrieves memory and handles cases where filters give no relevant prior query ids
            # I am going to do this by adding a condition whereby: 
            # 
            # If there are no `relevant_prior_query_ids`,THE MODEL WILL INHERITLY SET THE `naive_window` CONDITION TO BE TRUE, IN AN ATTEMPT TO MAINTAIN A SENSE OF CHAT CONTINUITY

            # if no relevant prior query ids were passed (as in, user does not want to restrict the window to particular id group, OR all potential ids were filtered out)
            if len(relevant_prior_query_ids) == 0 and naive_window != True:
                print("No relevant prior_query_ids passed, resorting to a naive_window for history")
                naive_window = True

            history_used_in_prompt_bool = True
            history_context.append("The numbers in the following lists indicate the chronological order of the queries (highest number is most recent):\n")
            
            # simple lookback based on chronological order of query states (history of query states)
            if naive_window:
                # get prior window size number of queries
                for i in range(max(1, current_query_num - window_size), current_query_num):
                    # Add previous user query with label
                    history_context.append(f"PRIOR QUERY {i}: {user_history[i]['query_text']}") 
                    
                    # Add previous model response with label, if it exists
                    if user_history[i]["response_text"]:                   
                        history_context.append(f"PRIOR RESPONSE {i}: {user_history[i]['response_text']}\n")

            # otherwise, we should utilized the relevant prior query ids to get filtered query / response pairs
            else:
                # filter the current query from the list of keys to loop over, pick the top `window_size` number of context chunks
                #all_history_state_id = list(user_history.keys())
                #prior_history_state_ids = sorted(all_history_state_id)[:-1]

                assert(current_query_num not in relevant_prior_query_ids)

                for prior_query_id in relevant_prior_query_ids:
                    history_context.append(f"PRIOR QUERY {prior_query_id}: {user_history[prior_query_id]['query_text']}")
                    history_context.append(f"PRIOR RESPONSE {prior_query_id}: {user_history[prior_query_id]['response_text']}\n")

        else:
            if window_size == 0:
                print("History was meant to be utilized, but the window_size is 0, so no lookback will be performed (window size == 0, no history requested).")

            if current_query_num > 1:
                print("History was meant to be utilized, but the query_number is 1, so no lookback can be performed (no history exists).")

    # Add current query with label to differentiate it
    #print(f"\n`build_historical_query_response_thread` for Query: {current_query_num} | History used : {history_used_in_prompt_bool} \n\n")

    history_context.append(f"\nCURRENT QUERY {current_query_num}: {user_history[current_query_num]['query_text']}")

    return "\n".join(history_context)

def standalone_prompt_model(llm_model, historical_prompt='', system_prompt='', context_prompt=''):

    full_prompt = system_prompt + historical_prompt + context_prompt

    #print(full_prompt)

    response = llm_model.generate_content(full_prompt)

    return response

def adjust_prompt(prompt, attempt):
    # Simple heuristic to bypass recitation filters
    return prompt + f"\n\n(Please respond in original phrasing.) [Retry {attempt}]"

# Had a wild error that needs to be handled.
# ValueError: Invalid operation: The `response.text` quick accessor requires the response to contain a valid `Part`, but none were returned. 
# The candidate's [finish_reason](https://ai.google.dev/api/generate-content#finishreason) is 4. Meaning that the model was reciting from copyrighted material.

from google.ai.generativelanguage_v1.types import Candidate

def generate_response_with_retry(prompt, model, max_tries=3):
    for attempt in range(1, max_tries + 1):
        try:
            response = model.generate_content(prompt)
            #print(f"[Attempt {attempt}] Response candidates:", response.candidates)

            if not response.candidates:
                print(f"⚠️ Attempt {attempt}: No candidates returned.")
                continue

            candidate = response.candidates[0]
            raw_reason = candidate.finish_reason                     # integer code
            reason = Candidate.FinishReason(raw_reason)             # enum wrapper

            # 1. RECITATION (unauthorized citation)
            if reason == Candidate.FinishReason.RECITATION:
                print(f"⚠️ Attempt {attempt}: RECITATION stop (code {raw_reason}).")
                prompt = adjust_prompt(prompt, attempt)
                continue

            # 2. SAFETY (content policy)
            elif reason == Candidate.FinishReason.SAFETY:
                print(f"⚠️ Attempt {attempt}: SAFETY filter (code {raw_reason}).")
                prompt = adjust_prompt(prompt, attempt)
                continue

            # 3. Normal STOP
            elif reason == Candidate.FinishReason.STOP:
                if candidate.content.parts:
                    print(f"✅ Attempt {attempt}: STOP with content (code {raw_reason}).\n")
                    return response
                else:
                    print(f"⚠️ Attempt {attempt}: STOP but no content to return.")
                    continue

            # 4. Anything else (MAX_TOKENS, BLOCKLIST, etc.)
            else:
                print(f"⚠️ Attempt {attempt}: Unhandled finish_reason {reason.name} (code {raw_reason}).")
                continue

        except Exception as e:
            # 5. API/client exceptions
            print(f"❌ Attempt {attempt}: Exception during generation: {type(e).__name__}: {e}")
            continue

    print("❌ All attempts failed. No valid response returned.\n")
    return None


def generate_response_with_retry_og(prompt, model, max_tries=3):
    for attempt in range(1, max_tries + 1):
        try:
            response = model.generate_content(prompt)
            print(response.candidates)

            # "response_text": response.text,
            # "prompt_token_count": response.usage_metadata.prompt_token_count,
            # "candidates_token_count": response.usage_metadata.candidates_token_count,
            # "total_token_count": response.usage_metadata.total_token_count,


            print(response)
            if not response.candidates:
                print(f"⚠️ Attempt {attempt}: No candidates returned.")
                continue

            candidate = response.candidates[0]
            finish_reason = candidate.finish_reason

            if finish_reason == "RECITATION":
                print(f"⚠️ Attempt {attempt}: Response blocked due to recitation.")
                prompt = adjust_prompt(prompt, attempt)
                continue
            elif finish_reason == "SAFETY":
                print(f"⚠️ Attempt {attempt}: Response blocked due to safety filters.")
                prompt = adjust_prompt(prompt, attempt)
                continue
            elif finish_reason == "STOP" and candidate.content.parts:
                print(f"✅ Attempt {attempt}: Response generated successfully.")
                return response  # Return full response object
            else:
                print(f"⚠️ Attempt {attempt}: Unhandled finish_reason: {finish_reason}.")


        except Exception as e:
            print(f"❌ Attempt {attempt}: Exception during generation: {e}")
    print("❌ All attempts failed. No valid response returned.")
    return None

def topic_level_context_retrieval(current_query_context_chunks, QDRANT_CLIENT, CHUNK_COLLECTION, verbose=True):

    # Get the context chunk ids for the current state
    current_state_context_chunk_ids = [chunk.id for chunk in current_query_context_chunks]

    # Compute the current state's average context score (sort of a general measurement, can be refined later)
    #current_state_context_scores = [chunk.score for chunk in current_query_context_chunks]

    # Get the arxiv ids for the current context chunk so that they can be included in the response
    #current_state_context_arxiv_titles = [chunk.payload['pdf_metadata']['title'] for chunk in current_query_context_chunks]
    current_state_context_arxiv_ids = [chunk.payload['pdf_metadata']['id'] for chunk in current_query_context_chunks]


    # Initialize a set to track (arxiv_id, header) pairs that have been seen
    topic_seen = set()

    # Initialize counters for new and duplicate topics
    new_topic_count = 0
    duplicate_topic_count = 0

    # Loop through the chunks and detect duplicates based on arxiv_id and header_metadata
    for chunk in current_query_context_chunks:
        chunk_id = chunk.id
        arxiv_id = chunk.payload['pdf_metadata']['id']
        header_metadata = chunk.payload['header_metadata']
        
        # Process the header values by iterating over all headers and considering them by their content
        relevant_headers = [value for key, value in header_metadata.items() if value]

        # Create a key based on arxiv_id and the full header level sequence (top to bottom)
        # The key will include all header levels to uniquely represent the topic
        topic_key = (arxiv_id, tuple(relevant_headers))  # Using tuple to store the full sequence of headers

        if topic_key in topic_seen:
            duplicate_topic_count += 1
            #print(f"#### At Chunk ID: {chunk_id}, Duplicate topic detected for arXiv ID {arxiv_id} and headers: {relevant_headers}")
        else:
            new_topic_count += 1
            topic_seen.add(topic_key)
            #print(f"#### At Chunk ID: {chunk_id}, New topic detected for arXiv ID {arxiv_id} and headers: {relevant_headers}")

    # This will print out duplicates and new topics as the chunks are processed

    if verbose:
        # Print counts at the end
        print(f"\n#### Total new topics detected: {new_topic_count}")
        print(f"#### Total duplicate topics detected: {duplicate_topic_count}")
        print(f'\n#### Topic Key: {topic_seen}')
        print('####\n')

    # Initialize the new dictionary structure
    structured_topic_dict = {}

    # Restructure the data
    for arxiv_id, header_tuple in topic_seen:
        # Check if arxiv_id is already a key in the dictionary
        if arxiv_id not in structured_topic_dict:
            structured_topic_dict[arxiv_id] = []
        
        # Append the header tuple to the list for this arxiv_id
        structured_topic_dict[arxiv_id].append(header_tuple)

    # Print the restructured dictionary
    #print('\n\nStructured Topic Dict: ', structured_topic_dict)

    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    def get_chunk_ids_by_arxiv(qdrant_client, collection_name, arxiv_id, headers=None):
        """
        Get chunk IDs by searching for the arxiv_id in Qdrant.
        """
        chunk_ids_by_header = {}

        pdf_id_filter = Filter(
                must=[
                    FieldCondition(
                        key="pdf_metadata.id",  # Filtering on the nested pdf_metadata['id']
                        match=MatchValue(value=arxiv_id)
                    )
                ]
            )

        # Perform the count query to get number of chunks that fall within a given arxiv_id
        arxiv_num_chunks = qdrant_client.count(collection_name=collection_name, 
                                                count_filter=pdf_id_filter
        )

        # Get all of the chunks associated with a single arxiv id
        response = qdrant_client.scroll(collection_name=collection_name, 
                                        scroll_filter=pdf_id_filter,
                                        limit=arxiv_num_chunks.count
        )

        #print('\n\n', response, '\n\n')
        #print('\n\n\n', response)

        # Extract the title from the first point in the response (same for all points)
        #title = response[0][0].payload['pdf_metadata']['title']
        
        # aggregate each point to a specific header; if it matches with header from the input header list, update the `chunk_ids_by_header` to reflect this
        for point in response[0]:

            temp_payload = point.payload
            temp_headers = temp_payload['header_metadata']

            # Flatten the dictionary into a list, excluding None values
            temp_flattened_headers = [value for value in temp_headers.values() if value is not None]

            # For each header in the flattened list
            for header in temp_flattened_headers:
                # Create a unique key for this header (title > section)
                header_key = header  # Using the header value as the key

                # If the header is not in the dictionary, initialize it
                if header_key not in chunk_ids_by_header:
                    chunk_ids_by_header[header_key] = []

                # Append the chunk ID to the corresponding header key
                chunk_ids_by_header[header_key].append(point.id)

        # chunk_ids_by_header format: {'1 Introduction': [8106, 8107], '2 Background': [8108, 8109], ...}                    

        # Remove the header that matches the title from the dictionary
        #if title in chunk_ids_by_header:
            #del chunk_ids_by_header[title]
        
        #print(f"\n\nChunk IDs for arXiv ID {arxiv_id}: {chunk_ids_by_header}\n\n")

        return chunk_ids_by_header        

    def get_chunk_ids_by_bottom_level_headers(chunk_ids_by_header, example_headers):
        """
        Get a dictionary of chunk IDs that match the bottom-level sections of the original header tuples.
        This function handles malformed or empty headers by assigning unique placeholders
        to preserve distinction and avoid key collisions in the dictionary.
        """
        # === Step 1: Safely extract bottom-level headers ===
        bottom_level_headers = []
        header_placeholder_counter = 0

        for header in example_headers:
            if len(header) > 0:
                bottom_level_headers.append(header[-1])
            else:
                placeholder = f"ERROR_NO_HEADER_PLACEHOLDER_{header_placeholder_counter}"
                #print(f"⚠ Assigned unique placeholder to bottom_level_headers: {placeholder}")
                bottom_level_headers.append(placeholder)
                header_placeholder_counter += 1

        #if header_placeholder_counter > 1:
            #print(f"!!! MULTIPLE_UNIQUE_PLACEHOLDERS_IN_HEADERS: {header_placeholder_counter} unknown example_headers assigned unique placeholders")

        # === Step 2: Normalize keys in chunk_ids_by_header ===
        normalized_chunk_ids_by_header = {}
        chunk_placeholder_counter = 0

        for key, value in chunk_ids_by_header.items():
            if key is None or (isinstance(key, str) and key.strip() == ''):
                placeholder_key = f"ERROR_NO_HEADER_PLACEHOLDER_{chunk_placeholder_counter}"
                #print(f"⚠ Assigned unique placeholder key to chunk_ids_by_header: {placeholder_key} for value: {value}")
                normalized_chunk_ids_by_header[placeholder_key] = value
                chunk_placeholder_counter += 1
            else:
                normalized_chunk_ids_by_header[key] = value

        #if chunk_placeholder_counter > 1:
            #print(f"!!! MULTIPLE_UNIQUE_PLACEHOLDERS_IN_CHUNKS: {chunk_placeholder_counter} malformed chunk_ids_by_header keys assigned unique placeholders")

        # === Step 3: Filter normalized chunk IDs by bottom-level headers ===
        filtered_chunk_ids_by_header = {}
        #print("\nFiltering normalized_chunk_ids_by_header based on bottom_level_headers...")
        #print("Bottom-level headers:", bottom_level_headers, "\n")

        for key, value in normalized_chunk_ids_by_header.items():
            if key in bottom_level_headers:
                #print(f"✔ Match found: {repr(key)} is in bottom_level_headers. Adding to result.")
                filtered_chunk_ids_by_header[key] = value
            #else:
                #print(f"✘ No match: {repr(key)} not in bottom_level_headers. Skipping.")

        #print("\nFiltered result keys:", list(filtered_chunk_ids_by_header.keys()))
        return filtered_chunk_ids_by_header

    # def get_chunk_ids_by_bottom_level_headers(chunk_ids_by_header, example_headers):
    #     """
    #     Get a list of chunk IDs that match the bottom-level sections of the original header tuples.
    #     This approach handles multi-level headers by always selecting the last level.
    #     """
    #     print('\n', type(example_headers))
    #     print("EXAMPLE_HEADERS:", example_headers, '\n\n')
    #     print("Chunk_IDS_BY_HEADER:", chunk_ids_by_header, '\n\n')

    #     # Initialize list for bottom-level headers
    #     bottom_level_headers = []

    #     # Safely extract bottom-level headers, handling empty tuples
    #     for header in example_headers:
    #         if len(header) > 0:
    #             bottom_level_headers.append(header[-1])
    #         else:
    #             bottom_level_headers.append("ERROR_NO_HEADER_PLACEHOLDER")

    #     # Diagnostic flag: check if the placeholder occurs multiple times
    #     placeholder_count = bottom_level_headers.count("ERROR_NO_HEADER_PLACEHOLDER")
    #     if placeholder_count > 1:
    #         print(f"!!! MULTIPLE_PLACEHOLDER_OCCURRENCES: 'ERROR_NO_HEADER_PLACEHOLDER' occurred {placeholder_count} times in bottom_level_headers")
        
    #     # Normalize chunk_ids_by_header to replace empty/null keys with a placeholder
    #     normalized_chunk_ids_by_header = {}
    #     for key, value in chunk_ids_by_header.items():
    #         normalized_key = key if (key is not None and key.strip() != '') else "ERROR_NO_HEADER_PLACEHOLDER"
    #         normalized_chunk_ids_by_header[normalized_key] = value

    #     # Filter the normalized dictionary keys based on the bottom-level headers
    #     filtered_chunk_ids_by_header = {}
    #     print("\nFiltering chunk_ids_by_header based on bottom_level_headers...")
    #     print("Bottom-level headers:", bottom_level_headers, "\n")

    #     print("Normalized Chunk_ids_by_header:")
    #     for key, value in normalized_chunk_ids_by_header.items():
    #         if key in bottom_level_headers:
    #             print(f"✔ Match found: {repr(key)} is in bottom_level_headers. Adding to result.")
    #             filtered_chunk_ids_by_header[key] = value
    #         else:
    #             print(f"✘ No match: {repr(key)} not in bottom_level_headers. Skipping.")

    #     print("\nFiltered result keys:", list(filtered_chunk_ids_by_header.keys()))

    #     # # Filter the dictionary keys based on the bottom-level headers
    #     # filtered_chunk_ids_by_header = {}
    #     # print("Filtering chunk_ids_by_header based on bottom_level_headers...")
    #     # print("Bottom-level headers:", bottom_level_headers, "\n")

    #     # print(f"Chunk_ids_by_header: {chunk_ids_by_header}\n")

    #     # for key, value in chunk_ids_by_header.items():
    #     #     if key in bottom_level_headers:
    #     #         print(f"✔ Match found: '{key}' is in bottom_level_headers. Adding to result.")
    #     #         filtered_chunk_ids_by_header[key] = value
    #     #     else:
    #     #         print(f"✘ No match: '{key}' not in bottom_level_headers. Skipping.")

    #     # print("\nFiltered result keys:", list(filtered_chunk_ids_by_header.keys()))

    #     # # Filter the dictionary keys based on the bottom-level headers
    #     # filtered_chunk_ids_by_header = {}
    #     # for key, value in chunk_ids_by_header.items():
    #     #     if key in bottom_level_headers:
    #     #         filtered_chunk_ids_by_header[key] = value

    #     # Check for presence of a bottom level header
    #     #if len(example_headers) >= 1:
    #     #    try:
    #     #        # Extract the bottom-level header from each original header tuple
    #     #        bottom_level_headers = [header[-1] for header in example_headers]
    #     #    except IndexError:
    #     #        print(example_headers)
    #     #else:
    #     #    # if no header, assign "ERROR_NO_HEADER_PLACEHOLDER"
    #     #    bottom_level_headers = ["ERROR_NO_HEADER_PLACEHOLDER" for _ in example_headers]
        
    #     # # Filter the dictionary keys based on the bottom-level headers
    #     # filtered_chunk_ids_by_header = {
    #     #     key: value
    #     #     for key, value in chunk_ids_by_header.items()
    #     #     if key in bottom_level_headers
    #     # }

    #     return filtered_chunk_ids_by_header

    def process_structure_dict_entries(input_structured_dict):

        original_chunk_id_to_id_pool_mapper = {}
        arxiv_id_mapper = {}
        
        for structured_dict_arxiv_id, example_headers in input_structured_dict.items():
            chunk_ids_by_header = get_chunk_ids_by_arxiv(qdrant_client=QDRANT_CLIENT, 
                                        collection_name=CHUNK_COLLECTION, 
                                        arxiv_id=structured_dict_arxiv_id, # this is a filter
                                        headers=example_headers
                                        )

            # Retrieve the chunk IDs that match the bottom-level headers
            filtered_chunk_ids_by_topic_header = get_chunk_ids_by_bottom_level_headers(chunk_ids_by_header, example_headers)

            #print(f"\n\nFinal Topic Chunk Ids by Header = {filtered_chunk_ids_by_topic_header}")
            #print()  


            # Relate chunk ids to their respective topic header
            filtered_chunk_id_to_topic = {}

            for topic, chunk_ids in filtered_chunk_ids_by_topic_header.items():
                for chunk_id in chunk_ids:
                    filtered_chunk_id_to_topic[chunk_id] = topic

            #print(f"\n\nChunk Id to Topic = {filtered_chunk_id_to_topic}")
            #print()
            

            # Output the mapping
            for original_chunk_id, original_arxiv_id in zip(current_state_context_chunk_ids, current_state_context_arxiv_ids):
                if structured_dict_arxiv_id == original_arxiv_id:
                    #print(original_chunk_id)
                    chunk_topic = filtered_chunk_id_to_topic[original_chunk_id]
                    chunk_pool = filtered_chunk_ids_by_topic_header[chunk_topic]

                    if verbose:
                        print(f"ArXiv ID {original_arxiv_id} Chunk ID: {original_chunk_id} is related to Topic: `{chunk_topic}` with Chunk ID pool: {chunk_pool}")

                    original_chunk_id_to_id_pool_mapper[original_chunk_id] = chunk_pool
                    arxiv_id_mapper[original_chunk_id] = {'arxiv_id': original_arxiv_id, 'topic': chunk_topic}

        return original_chunk_id_to_id_pool_mapper, arxiv_id_mapper

    single_chunk_to_topic_chunks_pool_mapper, arxiv_id_mapper = process_structure_dict_entries(input_structured_dict=structured_topic_dict)

    return single_chunk_to_topic_chunks_pool_mapper, arxiv_id_mapper

def display_summary(current_user_query_state_history: dict):
        print('####\n############### RESPONSE:')
        print('####')
        
        # Current state semantic chunk info
        print('#### 🔹 Current State Semantic Context Chunks:')
        semantic_context_ids = current_user_query_state_history.get("current_state_context_ids") or []
        semantic_context_scores = current_user_query_state_history.get("current_state_context_scores") or []
        
        if not semantic_context_ids or not semantic_context_scores:
            print("####     None")
        else:
            for i, (cid, score) in enumerate(zip(semantic_context_ids, semantic_context_scores)):
                print(f"####     [{i+1}] ID: {cid}, Score: {score:.4f}")

        # Current state bm25 chunk info
        print('\n#### 🔹 Current State BM25 Context Chunks:')
        bm25_context_ids = current_user_query_state_history.get("current_state_bm25_context_ids") or []
        bm25_context_scores = current_user_query_state_history.get("current_state_bm25_context_scores") or []
        
        if not bm25_context_ids or not bm25_context_scores:
            print("####     None")
        else:
            for i, (cid, score) in enumerate(zip(bm25_context_ids, bm25_context_scores)):
                print(f"####     [{i+1}] ID: {cid}, Score: {score:.4f}")

        # Current state hybrid-fused chunk info
        print('\n#### 🔹 Current State Actual Utilized Context Chunks:')

        utilized_ids = current_user_query_state_history.get("context_ids_utilized") or []
        sources     = current_user_query_state_history.get("context_ids_source")   or []
        fused_scores = current_user_query_state_history.get("current_state_hybrid_fused_scores") or []

        if not utilized_ids:
            print("####     None")
        else:
            for i, (cid, src, score) in enumerate(zip(utilized_ids, sources, fused_scores), start=1):
                print(f"####     [{i}] ID: {cid}, Source: {src}, Fused Score: {score:.4f}")

        # Prior state info (if HISTORY_SWITCH was used)
        considered_ids = current_user_query_state_history.get("considered_prior_state_ids", [])
        score_mask = current_user_query_state_history.get("similarity_score_mask", [])
        utilized_ids = current_user_query_state_history.get("utilized_prior_state_ids", [])

        print('\n#### 🔹 Prior State History Utilization:')
        print(f"####     Considered Prior State IDs     : {considered_ids}")
        print(f"####     Similarity Score Mask          : {score_mask}")
        print(f"####     Utilized Prior State IDs (pass): {utilized_ids}\n")

def get_system_prompt():
    my_system_prompt = """
    You are an advanced AI research assistant. Generate detailed and comprehensive responses that supplement students' and academic researchers' work with information grounded in highly cited AI/ML research papers, specifically in fields like NLP and CV. The response should not focus on one area of study but should be informed by both the current query and chat history to generate a well-rounded answer.

    1. **Introductory Overview**: Start with a high-level conceptual overview of the topic, providing a brief and clear explanation that covers the essential aspects of the subject. This should be accessible to a broad audience.

    2. **Technical Overview**: After the conceptual overview, provide a more in-depth, technical explanation that dives deeper into the topic. This could include relevant algorithms, methods, or models, as well as their theoretical foundations.

    3. **Example-Based Expansion**: Throughout the response, incorporate examples from relevant research to illustrate key concepts. These examples should come from generalized research trends and not focus on specific papers or studies, helping to broaden the context.

    4. **Broader Exploration**: After addressing the original query, provide suggestions for related topics or areas for further exploration, encouraging the user to expand their understanding. The exploration should relate to the current query and prior query/response pairs, offering natural extensions to the discussion, such as other approaches, applications, or advancements related to the topic.

    The tone should be professional yet approachable, offering a balance of conceptual clarity and technical depth. The response should not be overly simplistic, but should aim to make complex topics understandable while offering substantial detail. Use direct quotes where relevant, but focus primarily on summarizing findings from academic research.
    """
    print(f"\n############### System Prompt:\n{my_system_prompt}")

    return my_system_prompt

# Function to extract publish year and month from arxiv_id
def get_publish_date(arxiv_id):
    year_month_together = arxiv_id.split('.')[0]
    year, month = year_month_together[:2], year_month_together[3:]

    year = '20' + year
    if len(month) < 2:
        month = '0' + month

    return f"{year}-{month}"

def build_citation_string_final(chunk_info_display):
    lines = []
    for arxiv_id, title, chunk_id, header_hierarchy, token_count in chunk_info_display:
        publish_date = get_publish_date(arxiv_id)
        year = publish_date[:4] if publish_date else "Unknown"
        section_path = " → ".join(v for _, v in header_hierarchy.items())
        link = f"https://arxiv.org/abs/{arxiv_id}"

        line = (
            f"“{title}” ({year}) |  arXiv:{arxiv_id} | {link} | Section: {section_path} | "
            f"Tokens: {token_count} | Chunk {chunk_id}"
        )
        lines.append(line)
    return "\n".join(lines)

