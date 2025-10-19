from sentence_transformers import SentenceTransformer
from data.chroma_client import chroma_client

client = chroma_client.client
collection = chroma_client.get_collection("test")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

text = "Hello world"
embedding = embedder.encode(text)
print(embedding)
print(embedding.shape)

collection.add(
    documents=[text],
    embeddings=[embedding.tolist()],
    ids=["1"],
)

results = collection.query(
    query_texts=["Hello world"],
    n_results=1,
)
print(results)

chroma_client.delete_collection("test")
