import json
import pickle
import os

def save_chat_json(query_state, file_path='user_output/query_data.json'):
    # Check if the file already exists
    if os.path.exists(file_path):
        # Load existing data
        with open(file_path, 'r') as f:
            data = json.load(f)
        if "query_states" not in data:
            data["query_states"] = []
    else:
        # Initialize new structure
        data = {"query_states": []}

    # Remove embedding if present
    query_state_no_embed = {k: v for k, v in query_state.items() if k != "query_embedding"}

    # Append the new query state
    data["query_states"].append(query_state_no_embed)

    # Save back to the file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_chat_pkl(file_path="user_output/user_history.pkl"):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        query_states = data.get("query_states", [])
        # Convert to dict keyed by query_number for easier access
        return {int(q["query_number"]): q for q in query_states}
    else:
        # No file exists yet — return an empty history
        return {}

def load_chat_json(file_path="user_output/query_data.json"):
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        query_states = data.get("query_states", [])
        # Convert to dict keyed by query_number for easier access
        return {int(q["query_number"]): q for q in query_states}
    else:
        # No file exists yet — return an empty history
        return {}

    # Export the collection to .pkl format for storage as a file

def save_all_chat_pkl_by_embedding(user_query_state_history,
                                   embedded_path='user_output/user_embedded_history.pkl',
                                   non_embedded_path='user_output/user_non_embedded_history.pkl'):
    # Filter into two separate lists
    embedded = [v for v in user_query_state_history.values() if v.get("rag_used")]
    non_embedded = [v for v in user_query_state_history.values() if not v.get("rag_used")]

    with open(embedded_path, 'wb') as f:
        pickle.dump({"query_states": embedded}, f)

    with open(non_embedded_path, 'wb') as f:
        pickle.dump({"query_states": non_embedded}, f)

def save_chat_pkl_by_embedding(user_query_state_history,
                                embedded_path='user_output/user_embedded_history.pkl',
                                non_embedded_path='user_output/user_non_embedded_history.pkl'):
    # Load or initialize existing embedded history
    if os.path.exists(embedded_path):
        with open(embedded_path, 'rb') as f:
            embedded_data = pickle.load(f)
    else:
        embedded_data = {"query_states": []}

    # Load or initialize existing non-embedded history
    if os.path.exists(non_embedded_path):
        with open(non_embedded_path, 'rb') as f:
            non_embedded_data = pickle.load(f)
    else:
        non_embedded_data = {"query_states": []}

    # Create lookup sets for already saved entries
    embedded_ids = {entry["query_number"] for entry in embedded_data["query_states"]}
    non_embedded_ids = {entry["query_number"] for entry in non_embedded_data["query_states"]}

    # Split and append only new entries
    for q_num, entry in user_query_state_history.items():
        if entry.get("rag_used") and q_num not in embedded_ids:
            embedded_data["query_states"].append(entry)
        elif not entry.get("rag_used") and q_num not in non_embedded_ids:
            non_embedded_data["query_states"].append(entry)

    # Save back to files
    with open(embedded_path, 'wb') as f:
        pickle.dump(embedded_data, f)

    with open(non_embedded_path, 'wb') as f:
        pickle.dump(non_embedded_data, f)
