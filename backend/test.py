from app import user_rag_query

job_1 = {
    'USER_INPUT_QUERY' : 'hello, how are you today?',
    'DESIRED_HISTORY_WINDOW_SIZE' : 3,
    'DESIRED_CONTEXT_CHUNKS_TOP_K' : 5,
    'RAG_SWITCH' : True,
    'HISTORY_SWITCH' : True,
    'BM25_SWITCH' : True,
    'TOPIC_RETRIEVAL_SWITCH' : True,
    'HISTORIC_QUERY_SIMILARITY_THRESHOLD' : 0.3,
}

print(user_rag_query("User_1", "Chat_1", job_1))
print(user_rag_query("User_1", "Chat_1", job_1))