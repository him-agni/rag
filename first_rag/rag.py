# Step 1: Load the knowledge base
with open("my_knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

print(text)
print("Characters loaded:", len(text))

# Step 2: Split the text into chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=190,
    chunk_overlap=20,
)
chunks = splitter.split_text(text)

print("\nNumber of chunks:", len(chunks))
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ({len(chunk)} chars) ---")
    print(chunk)
print("\nOverlap check:")
for i in range(len(chunks) - 1):
    a, b = chunks[i], chunks[i + 1]
    shared = ""
    for n in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:n]):
            shared = b[:n]
            break
    print(f"chunk {i} -> chunk {i+1}: {shared!r}")

# Step 3: Turn chunks into embeddings
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

print("\nEmbeddings shape:", embeddings.shape)
print("First 5 numbers of chunk 0:", embeddings[0][:5])
question = "How many vacation days do I get?"
q_emb = model.encode(question)
scores = model.similarity(q_emb, embeddings)[0]

for chunk, score in zip(chunks, scores):
    print(f"{score:.3f}  {chunk[:50]!r}")

# Step 4: Store the embeddings in a FAISS index
import faiss
import numpy as np

embeddings = np.array(embeddings).astype("float32")
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("\nVectors in index:", index.ntotal)

question = "How many vacation days do I get?"
q_emb = model.encode([question]).astype("float32")

distances, ids = index.search(q_emb, 2)
print("Closest chunk ids:", ids[0])
print("Distances:", distances[0])
for i in ids[0]:
    print("->", chunks[i][:60])
