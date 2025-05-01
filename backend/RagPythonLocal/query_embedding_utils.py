import torch
from qdrant_client import QdrantClient

def get_query_embedding(model, tokenizer, query_text='', device='cpu'):
    query_tokens = tokenizer(query_text, padding=True, truncation=True, return_tensors='pt', max_length=8192)
    query_tokens = {k: v.to(device) for k, v in query_tokens.items()}

    with torch.no_grad():
        query_embedding = model(**query_tokens)[0][:, 0]

    norm_query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)

    return norm_query_embedding.cpu().numpy().tolist()[0]

# Gets closest similarity score chunks
def get_context_chunks(qdrant_client=QdrantClient, norm_query_embedding=[], num_chunks=5, filters=None, my_collection='search_collection'):

    context_chunks = qdrant_client.query_points(
                      collection_name=my_collection,
                      query=norm_query_embedding,
                      limit=num_chunks,
                      query_filter=filters
    ).points

    return context_chunks