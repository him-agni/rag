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

# Step 5: Retrieve, Augment, Generate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
generator = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
MAX_DISTANCE = 1.4


def retrieve(question, k=2):
    q_emb = model.encode([question]).astype("float32")
    distances, ids = index.search(q_emb, k)
    return [chunks[i] for i in ids[0]], distances[0]


def augment(question, retrieved_chunks):
    context = "\n\n".join(retrieved_chunks)
    return (
        "Answer the question using only the context below. "
        "If the answer is not in the context, reply exactly: I don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    output_ids = generator.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def ask(question, k=2):
    retrieved, distances = retrieve(question, k)
    print("\nQuestion:", question)
    print("Distances:", [round(float(d), 3) for d in distances])

    # Threshold: keep only chunks close enough to the question
    relevant = [c for c, d in zip(retrieved, distances) if d <= MAX_DISTANCE]
    if not relevant:
        print("Answer: I don't know (no relevant chunk found)")
        return

    print("Retrieved:", [c[:40] for c in relevant])
    print("Answer:", generate(augment(question, relevant)))


ask("Is there free lunch?")

