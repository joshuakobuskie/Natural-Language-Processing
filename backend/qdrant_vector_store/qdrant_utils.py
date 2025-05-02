# Standard Library
import os
import json
import pickle

# Qdrant Client
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Batch
from qdrant_client.models import Distance, VectorParams

# Function to upsert datset (with embeddings) to qdrant client in order to save in qdrant expected format before saving to disk
def upsert_dataset_to_qdrant_client(dataset, collection_name=str, qdrant_client=QdrantClient):
    points = []

    for idx, example in enumerate(dataset):
        vector = example["embedding"]
        if vector is None:
            print(f"Skipping index {idx}: missing embedding.")
            continue

        point = PointStruct(
            id=idx,
            vector=vector,
            payload={
                "text": example["markdown_text"],
                "pdf_metadata": example["pdf_metadata"],
                "header_metadata": example["header_metadata"],
                "chunk_metadata": example["chunk_metadata"],
            }
        )
        points.append(point)

    # Upload all points at once
    qdrant_client.upsert(collection_name=collection_name, points=points)

    # Retrieve the number of points (vectors) in the collection (if we expand dataset, this may need to change)
    num_points = qdrant_client.count(collection_name=collection_name).count

    print(f"\nUpserted {num_points} / {len(points)} points to Qdrant collection: {collection_name}.")

# Export a collection from the QdrantClient
def export_qdrant_vectors(collection_name=str, qdrant_client=QdrantClient, folder_path='qdrant_vector_store', output_file="qdrant_vectors.pkl"):

    file_type = output_file.split('.')[-1]
    #print(file_type)
    if file_type != 'json' and file_type != 'pkl':
        # Failed to save
        print("extension must be .pkl or .json")
        return

    # Retrieve the number of points (vectors) in the collection (if we expand dataset, this may need to change)
    num_points = qdrant_client.count(collection_name=collection_name).count

    # Retrieve all points (vectors + metadata)
    all_points = qdrant_client.scroll(collection_name=collection_name, scroll_filter=None, with_vectors=True, limit=num_points)[0]

    # Convert to dictionary format for JSON
    export_data = [
        {
            "id": point.id,
            "vector": point.vector,
            "payload": point.payload
        }
        for point in all_points
    ]

    export_filepath = os.path.join(folder_path, output_file)

    if file_type == 'json':
        # Save to JSON file
        with open(export_filepath, "w") as f:
            json.dump(export_data, f, indent=4)
    elif file_type == 'pkl':
        # Save using Pickle
        with open(export_filepath, "wb") as f:
            pickle.dump(all_points, f)

    print(f"Exported {len(all_points)} / {num_points} qdrant points with vectors to {export_filepath}\n")

def load_to_vector_store(qdrant_client=QdrantClient, collection_name=str, from_pkl=True, pkl_filepath="qdrant_vector_store/qdrant_vectors.pkl"):

    #if not isinstance(qdrant_client, QdrantClient):
    #    raise ValueError("qdrant_client must be an instance of QdrantClient")

    if from_pkl:
        # Load the .pkl file
        with open(pkl_filepath, "rb") as f:
            data = pickle.load(f)  # This should be a list of dicts with keys: "id", "vector", "payload"

        #print(len(data))

        # Extract ids, vectors, and payloads from Record objects
        ids = [record.id for record in data]
        vectors = [record.vector for record in data]
        payloads = [record.payload for record in data]

    # Create a Batch object
    my_batch = Batch(ids=ids, vectors=vectors, payloads=payloads)

    qdrant_client.upsert(
        collection_name=collection_name,
        points=my_batch
    )

    print(f"Upserted batch of {len(ids)} points to qdrant client collection {collection_name}")

def upsert_history_to_vector_store(history: dict = None, qdrant_client: QdrantClient = None, collection_name: str = "", batch: bool = False):
    if not history:
        print("No history passed to upsert.")
        return {}

    if batch:
        ids = []
        vectors = []
        payloads = []

        for query_state in history.values():
            if query_state.get("query_embedding") is None:
                continue  # skip entries without embeddings

            ids.append(query_state["query_number"])
            vectors.append(query_state["query_embedding"])

            payloads.append(build_query_state_payload(query_state))

        # Efficient batch upload
        qdrant_client.upsert(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=payloads)
        )

    else:
        point_id = history["query_number"]
        query_embedding = history["query_embedding"]
        
        payload = build_query_state_payload(history)

        my_point = PointStruct(
            id=point_id,
            vector=query_embedding,
            payload=payload
        )
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[my_point]
        )
        
        #print(f"Query #{point_id} point upserted to Qdrant collection: {collection_name} with ID: {point_id}\n")

def build_query_state_payload(query_state: dict) -> dict: # this function looks like database schema 
    return {
        "query_number": query_state.get("query_number"),
        "query_text": query_state.get("query_text"),
        #"query_embedding": query_state.get("query_embedding"), # commented out because we are retrieving payload state for qdrant upsert

        # Context from current state
        "current_state_context_ids": query_state.get("current_state_context_ids"),
        "current_state_context_scores": query_state.get("current_state_context_scores"),
        "current_state_context_average_score": query_state.get("current_state_context_average_score"),

        # Retrieval metadata
        "retrieval_top_k": query_state.get("retrieval_top_k"),
        "retrieval_method": query_state.get("retrieval_method"),
        "retrieval_filter_ids": query_state.get("retrieval_filter_ids"),
        "avg_similarity_to_context": query_state.get("avg_similarity_to_context"),
        "max_similarity_to_context": query_state.get("max_similarity_to_context"),
        "top_context_id": query_state.get("top_context_id"),
        "top_context_score": query_state.get("top_context_score"),

        # Prior state info (updated)
        "considered_prior_state_ids": query_state.get("considered_prior_state_ids"),
        "filter_similarity_score_threshold" : query_state.get("filter_similarity_score_threshold"),
        "filter_similarity_score_mask": query_state.get("filter_similarity_score_mask"),
        "utilized_prior_state_ids": query_state.get("utilized_prior_state_ids"),

        # Response info
        "response_text": query_state.get("response_text"),
        "prompt_token_count": query_state.get("prompt_token_count"),
        "candidates_token_count": query_state.get("candidates_token_count"),
        "total_token_count": query_state.get("total_token_count"),
        "system_prompt_used": query_state.get("system_prompt_used"), # the initial system prompt used to generate the current response
        "dynamic_prompt_body": query_state.get("dynamic_prompt_body"), # the dynamic body prompt generated through the `build_historical_query_response_thread` function

        # Flags
        "rag_used": query_state.get("rag_used"),
        "history_used": query_state.get("history_used"),
        "rag_and_history_used": query_state.get("rag_and_history_used"),
        "desired_lookback_window_size": query_state.get("desired_lookback_window_size"), # the desired input lookback window size
        "actual_lookback_window_size": query_state.get("actual_lookback_window_size"), # the size of the actual lookback window after taking into account history length
        
        # Timing
        "query_start_time": query_state.get("query_start_time"),
        "query_finish_time": query_state.get("query_finish_time"),
        "query_processing_time": query_state.get("query_processing_time"),

        # Experiment Flag
        "experiment_tag": query_state.get("experiment_tag"),
        
        ##############
        # Additions for Fused Context Info
        ##############
        "context_ids_utilized": query_state.get("context_ids_utilized"),
        "context_ids_source": query_state.get("context_ids_source"),
        "current_state_hybrid_fused_scores": query_state.get("current_state_hybrid_fused_scores"),

        ##############
        # Additions for BM25 Specific Info
        ##############
        "bm25_RRF_constant": query_state.get("bm25_RRF_constant"),  # For RRF constant
        "bm25_multiplier": query_state.get("bm25_multiplier"), # Chunk Multiplier for hybrid retrieval
        "current_state_bm25_context_ids": query_state.get("current_state_bm25_context_ids"),
        "current_state_bm25_context_scores": query_state.get("current_state_bm25_context_scores"),

        ##############
        # Additions for Topic Level Retrieval Info (if available)
        ##############
        "topic_retrieval_used": query_state.get("topic_retrieval_used"),  # bool: Whether topic-level retrieval was used
        "top_topic_chunk_ids": query_state.get("top_topic_chunk_ids"),  # The selected top topic chunk IDs retrieved
        "top_topic_chunk_similarity": query_state.get("top_topic_chunk_similarity"),  # The similarity scores for the selected topic chunks
        "weight_coherence": query_state.get("weight_coherence"),  # Weight for internal topic coherence (higher is better)
        "weight_query_relevance": query_state.get("weight_query_relevance"),  # Weight for how well the topic matches the query
        "weight_chunk_count": query_state.get("weight_chunk_count"),  # Weight for the number of chunks in the topic (higher is better)
    
    }

def instantiate_collection(qdrant_client=QdrantClient, collection_name='search_collection', embedding_size=1024):

    if not isinstance(qdrant_client, QdrantClient):
        raise ValueError("qdrant_client must be an instance of QdrantClient")

    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=embedding_size, # Size of Snowflake Embedding Dimensions
                                    distance=Distance.COSINE), # Cosine similarity for vector search
    )

    #collection_info = qdrant_client.get_collection(collection_name=collection_name)

    #print(collection_info)

# LEGACY / UNUSED

def upsert_history_to_vector_store2(qdrant_client=QdrantClient, collection_name=str, history={}, batch=False):
    if batch:
        ids = []
        vectors = []
        payloads = []

        for query_state in history.values():
            if query_state.get("query_embedding") is None:
                continue  # skip entries without embeddings

            ids.append(query_state["query_number"])
            vectors.append(query_state["query_embedding"])

            payloads.append({
                "query_number": query_state.get("query_number"),
                "query_text": query_state.get("query_text"),
                "current_state_context_ids": query_state.get("current_state_context_ids"),
                "current_state_context_average_score": query_state.get("current_state_context_average_score"),
                "prior_state_context_ids": query_state.get("prior_state_context_ids"),
                "prior_state_history_mask": query_state.get("prior_state_history_mask"),
                "response_text": query_state.get("response_text"),
                "prompt_token_count": query_state.get("prompt_token_count"),
                "candidates_token_count": query_state.get("candidates_token_count"),
                "total_token_count": query_state.get("total_token_count"),
                "rag_used": query_state.get("rag_used"),
                "history_used": query_state.get("history_used"),
                "query_start_time" : query_state.get("query_start_time"),
                "query_finish_time" : query_state.get("query_finish_time"),
                "query_processing_time" : query_state.get("query_processing_time")
            })

        # Efficient batch upload
        qdrant_client.upsert(
            collection_name=collection_name,
            points=Batch(ids=ids, vectors=vectors, payloads=payloads)
        )

    else:
        point_id = history["query_number"]
        query_embedding = history["query_embedding"]
        
        payload = {
            "query_number": history.get("query_number"),
            "query_text": history.get("query_text"),
            "current_state_context_ids": history.get("current_state_context_ids"),
            "current_state_context_average_score" : history.get("current_state_context_average_score"),
            "prior_state_context_ids" : history.get("prior_state_context_ids"),
            "prior_state_history_mask" : history.get("prior_state_history_mask"),
            "response_text": history.get("response_text"),
            "prompt_token_count": history.get("prompt_token_count"),
            "candidates_token_count" : history.get("candidates_token_count"),
            "total_token_count" : history.get("total_token_count"),
            "rag_used": history.get("rag_used"),
            "history_used" : history.get("history_used"),
            "query_start_time" : query_state.get("query_start_time"),
            "query_finish_time" : query_state.get("query_finish_time"),
            "query_processing_time" : query_state.get("query_processing_time")
        }

        my_point = PointStruct(
            id=point_id,
            vector=query_embedding,
            payload=payload
        )
        
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[my_point]
        )
        
        #print(f"Query #{point_id} point upserted to Qdrant collection: {collection_name} with ID: {point_id}\n")

def initialize_BM25(pkl_filepath=''):
    import pickle
    with open(pkl_filepath, "rb") as f:
        vector_data = pickle.load(f)  # This should be a list of dicts with keys: "id", "vector", "payload"
    
    chunk_payloads = [record.payload for record in vector_data]
    chunk_full_text = [chunk['text'] for chunk in chunk_payloads]

    import nltk
    # Preprocess: tokenize the documents using NLTK's word_tokenize
    tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in chunk_full_text]

    from rank_bm25 import BM25Okapi
    # Initialize BM25 on the tokenized corpus
    bm25_search_function = BM25Okapi(tokenized_corpus)
    
    return bm25_search_function