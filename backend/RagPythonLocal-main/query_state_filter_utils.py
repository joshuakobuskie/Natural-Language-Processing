# Standard Libraries
# Qdrant_client
from qdrant_client import QdrantClient # Qdrant Vector Database package
from qdrant_client.models import Filter, FieldCondition, Range, HasIdCondition

# For loading / using the embedding model and tokenizer
from query_embedding_utils import get_context_chunks

def current_query_to_prior_queries_filter(qdrant_client=QdrantClient,
                                        norm_query_embedding=[],
                                        num_chunks=5,
                                        my_collection='',
                                        filter_ids_on=[]
                                        ):
    
    # We have to apply a filter to a semantic similarity search that restricts the historical id filter to the last `desired_history_window_size` number of states
    historical_id_filter = Filter(
            must=[
                HasIdCondition(
                    has_id=filter_ids_on
                    )
                ]
    )

    # We apply the query to query semantic search to return the most similar queries of the prior 0 - 5 history states
    query_semantic_searched_history = get_context_chunks(qdrant_client=qdrant_client, 
                                                        norm_query_embedding=norm_query_embedding, 
                                                        num_chunks=num_chunks,
                                                        my_collection=my_collection,
                                                        filters=historical_id_filter
    )

    # Return as list of booleans preserving filter_ids_on order
    return query_semantic_searched_history


############# Probably should add a choice between `Naive_Context_Window` and `Filtered_Context_Window` #############
# `Naive_Context_Window` : Unfiltered History where all response query pairs are utilized to generate responses
# `Filtered_Context_Winow` : Filtered History method (lightweight/low latency) that eliminates semantically dis-similar
#  query-response pairs through various filter methods which allows for longer context window searches
#  
# Higher Filter Levels correspond with higher latency, but also more potential for differentiation between topic shifts and
# vague continuations, better long range topic callback detection, etc. The goal of filters is to try to make sure that the LLM
# is only being fed the most relevant context given the current query.
#
# Filter #1 - Query to Query Similarity Thresholding:
# This method compares the vector embeddings of queries against each other, and uses a threshold to decide which response_ids to include
# as query-response pairs when we pass our prompt to the LLM.
# 
# INPUT MORE FILTER DESCRIPTIONS HERE:
# 
# Preface - Higher filter levels will likely require more fine tuning.
# 
# Default Filter Level 0 : Query to Query Similarity thresholding (Filter 1 only)
# Filter Level 1 : Query to Query Similarity thresholding + Query-Context Cross Search (Filters 2, 3)
# Filter Level 2 : Query to Query Similarity thresholding + Query-Context Cross Search (Filters 2, 3) + Averaging Methods (Filters 4, 5)
#####################################################################################################################



"""
def current_query_to_prior_context_filter():

    id_list = []
    # We get a subset of valid chunks based on the past `actual_history_window_size` queries
    for prior_state_id in actual_history_id_window:
        id_list.extend(new_user_history[prior_state_id]['current_state_context_ids'])

    #id_list = list(set(id_list))
    # Deduplicate while preserving order
    id_list = list(dict.fromkeys(id_list))
    print("Chunks to Search: ", id_list)

    unique_id_prior_state_chunk_filter = Filter(
            must=[
                HasIdCondition(
                    has_id=id_list
                    )
                ]
    )

    # We apply the query to query semantic search to return the most similar queries of the prior 0 - 5 history states
    query_semantic_searched_history = get_context_chunks(qdrant_client=QDRANT_CLIENT, 
                                                        norm_query_embedding=user_text_query_embeddings, 
                                                        num_chunks=len(id_list), # I want a score for every chunk
                                                        my_collection=CHUNK_COLLECTION,
                                                        filters=unique_id_prior_state_chunk_filter
    )

    #print('Semantic Searched History:\n', query_semantic_searched_history)
    print('#################\n')

    final_valid_query_ids = []

    for prior_chunk in query_semantic_searched_history:
        if prior_chunk.score >= 0.4:
            final_valid_query_ids.append(prior_chunk.id)

    print(f"{len(final_valid_query_ids)} / {len(id_list)}")

return sorted(final_valid_query_ids)

def current_context_to_prior_queries_filter():

    #query_list = []
    # We get a subset of valid chunks based on the past `actual_history_window_size` queries
    #for prior_state_id in actual_history_id_window:
    #    query_list.extend(new_user_history[prior_state_id]['query_text'])

    #id_list = list(set(id_list))
    # Deduplicate while preserving order
    #id_list = list(dict.fromkeys(id_list))
    #print("Chunks to Search: ", id_list)

    unique_id_prior_state_chunk_filter = Filter(
            must=[
                HasIdCondition(
                    has_id=current_state_context_chunk_ids
                    )
                ]
    )

    all_prior_query_semantic_searched_history = {}

    for prior_state_id in actual_history_id_window:
        current_query = new_user_history[prior_state_id]['query_embedding']

        # We apply the query to query semantic search to return the most similar queries of the prior 0 - 5 history states
        prior_query_semantic_searched_history = get_context_chunks(qdrant_client=QDRANT_CLIENT, 
                                                            norm_query_embedding=current_query, 
                                                            num_chunks=len(current_state_context_chunk_ids), # I want a score for every chunk
                                                            my_collection=CHUNK_COLLECTION,
                                                            filters=unique_id_prior_state_chunk_filter
        )

        all_prior_query_semantic_searched_history[prior_state_id] = prior_query_semantic_searched_history

    #print('Semantic Searched History:\n', all_prior_query_semantic_searched_history)
    #print('#################\n')

    #final_valid_query_ids = []

    #for prior_chunk in query_semantic_searched_history:
    #    if prior_chunk.score >= 0.4:
    #        final_valid_query_ids.append(prior_chunk.id)

    #print(f"{len(final_valid_query_ids)} / {len(id_list)}")

return sorted(all_prior_query_semantic_searched_history)

def filter_4_avg_context_similarity(threshold=0.4):
    #current_query_context_chunks
    #print(current_query_context_chunks)
    current_context_embeddings_ids = [chunk.id for chunk in current_query_context_chunks]

    current_context_embeddings = QDRANT_CLIENT.retrieve(
                                                collection_name=CHUNK_COLLECTION,
                                                ids=current_context_embeddings_ids,
                                                with_vectors=True
                                            )#[0].vector

    current_context_embeddings = [chunk.vector for chunk in current_context_embeddings]

    top_prior_state_embeddings = []
    for prior_state_id in actual_history_id_window:
        temp_prior_context_id = new_user_history[prior_state_id]['current_state_context_ids'][:2] # this zero assumes the values are stored in order of highest similarity score, which they currently are
        if isinstance(temp_prior_context_id, int):
            temp_prior_context_id = list(temp_prior_context_id)
        
        #print(temp_prior_context_id)
        top_prior_state_embeddings.extend(QDRANT_CLIENT.retrieve(
                                                collection_name=CHUNK_COLLECTION,
                                                ids=temp_prior_context_id,
                                                with_vectors=True
                                            )
        )

        #print(top_prior_state_embeddings)

    top_prior_state_embeddings = [chunk.vector for chunk in top_prior_state_embeddings]

    if not current_context_embeddings or not top_prior_state_embeddings:
        print("Insufficient data for Filter 4.")
        return False  # Default to treating as continuation

    # Step 3: Compute full pairwise similarity matrix

    #print(current_context_embeddings, '\n')
    #print(top_prior_state_embeddings, '\n')

    sim_matrix = cosine_similarity(current_context_embeddings, top_prior_state_embeddings)
    avg_sim = np.mean(sim_matrix)

    print(f"[Filter 4] Average similarity to prior top chunks: {avg_sim:.3f} if below {threshold} False")

return avg_sim >= threshold

def filter_5_pairwise_query_drift_vs_current_baseline(threshold_margin=0.05):
    similarity_deltas = []

    # Fetch current context chunk vectors
    current_chunk_vectors = [
        chunk.vector for chunk in QDRANT_CLIENT.retrieve(
            collection_name=CHUNK_COLLECTION,
            ids=current_state_context_chunk_ids,
            with_vectors=True
        )
    ]

    # Historical baseline: similarity of *original* query embedding to current context
    current_baseline_score = new_user_history[query_num]['current_state_context_average_score']

    for prior_state_id in actual_history_id_window:
        prior_query_vector = new_user_history[prior_state_id]['query_embedding']

        # 1. Create pair-averaged embedding
        avg_pair_vector = np.mean([prior_query_vector, user_text_query_embeddings], axis=0)

        # 2. Similarity of averaged embedding to current context chunks
        sim_to_current = cosine_similarity([avg_pair_vector], current_chunk_vectors)[0].mean()

        # 3. Compare to original baseline similarity
        delta = sim_to_current - current_baseline_score
        similarity_deltas.append(delta)

        print(
            f"[Filter 5 - vs Current] Prior ID: {prior_state_id} | "
            f"Averaged→Current Δ = {delta:.3f} | Avg Pair Sim = {sim_to_current:.3f} | "
            f"Current Baseline = {current_baseline_score:.3f}"
        )

    avg_delta = np.mean(similarity_deltas)
    print(f"\n[Filter 5 - vs Current] Avg Δ Similarity across pairings: {avg_delta:.3f}")

return avg_delta >= threshold_margin

def filter_5_pairwise_query_drift(threshold=0.4):
    similarity_deltas = []

    for prior_state_id in actual_history_id_window:
        prior_query_vector = new_user_history[prior_state_id]['query_embedding']

        # 1. Create pair-averaged embedding
        avg_pair_vector = np.mean([prior_query_vector, user_text_query_embeddings], axis=0)

        # 2. Get top prior context chunk
        top_prior_context_id = new_user_history[prior_state_id]['current_state_context_ids'][0]
        
        prior_chunk_vector = QDRANT_CLIENT.retrieve(
            collection_name=CHUNK_COLLECTION,
            ids=[top_prior_context_id],
            with_vectors=True
        )[0].vector

        # 3. Get current context chunk vectors
        current_chunk_vectors = [
            chunk.vector for chunk in QDRANT_CLIENT.retrieve(
                collection_name=CHUNK_COLLECTION,
                ids=current_state_context_chunk_ids,
                with_vectors=True
            )
        ]

        # 4. Compute similarity to prior and current
        sim_to_prior = cosine_similarity([avg_pair_vector], [prior_chunk_vector])[0][0]
        sim_to_current = cosine_similarity([avg_pair_vector], current_chunk_vectors)[0].mean()

        delta = sim_to_current - sim_to_prior
        similarity_deltas.append(delta)

        print(f"[Filter 5 - Pair Avg] Prior ID: {prior_state_id} | Δ Similarity = {delta:.3f} (Current - Prior)")

    # Step 5: Evaluate whether average delta implies drift
    avg_delta = np.mean(similarity_deltas)
    print(f"\n[Filter 5 - Pair Avg] Avg Δ Similarity across pairs: {avg_delta:.3f}")

    # If similarity to current is consistently lower, it's a shift
return avg_delta >= -0.08
"""