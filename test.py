from data.chroma_client import ChromaClient

client1 = ChromaClient()
client2 = ChromaClient()

print(f"client 1 is client 2 {client1 is client2}" )
print(f"client 1 id {id(client1)}")
print(f"client 2 id {id(client2)}")



    