# RAG study notes (concepts)

Companion to progress_notes.md (which holds results). This file holds explanations. Updated as we go.

## 1. What is RAG?
An LLM only knows what it was trained on. RAG (Retrieval-Augmented Generation) gives it your documents at question time:
1. Retrieve - find the chunks of your documents relevant to the question
2. Augment - paste those chunks into the prompt with the question
3. Generate - the LLM answers using only that context

Prepare once: text file -> chunks -> vectors -> vector store (FAISS)
At question time: question -> vector -> nearest chunks -> prompt -> LLM -> answer

## 2. Tools and cost (all free, no API keys)
- sentence-transformers + all-MiniLM-L6-v2: text -> vectors (runs locally)
- faiss-cpu: vector storage and search
- langchain-text-splitters: chunking
- transformers + google/flan-t5-small: answer generation (runs locally)
Models download once from Hugging Face (no account needed).

## 3. Chunking
Why chunk: documents are too big for a prompt, and small focused chunks give clearer embeddings and less noise in the prompt.

RecursiveCharacterTextSplitter (rule-based, NO AI model):
- chunk_size is a MAXIMUM, not a target. Chunk sizes vary because the splitter cuts at natural boundaries.
- It tries separators in order: "\n\n" -> "\n" -> " " -> single characters. A piece is only split further if it is still too big.
- Small neighbouring pieces are merged back together while they fit.
- chunk_overlap repeats up to N characters from the end of one chunk at the start of the next, but ONLY where text was cut mid-piece. At chunk_size=190 each section fit whole, so there was no overlap.

Size trade-off:
- Too small: sentences cut, chunks lose context (e.g. "Friday are optional..." with no topic)
- Too big: several topics blur into one embedding; prompt fills with noise
- Real-world start: ~500-1000 chars, 10-20% overlap
- Limits to respect: embedding model truncates input (MiniLM ~256 tokens); generator has a small input window (question + k chunks must fit)
- Best method: measure. Write test questions with known answers, try several sizes, count how often the right chunk is in the top results (hit rate).

Techniques: fixed-size, recursive (ours), sentence-based, structure-based (headings), semantic (embedding-based).
Choose by data type:
- prose -> recursive; manuals/Markdown/HTML -> split by headings, keep heading with content
- code -> by function/class; tables -> by row with column names; FAQs -> one Q+A per chunk
- transcripts -> by speaker turn; legal/scientific -> by section
Tip: attach the section title to each chunk so it keeps its context.

Our file: chunk_size=190, overlap=20 gives 4 chunks, one per section.

## 4. Embeddings
- An embedding is a list of numbers (384 for all-MiniLM-L6-v2) representing the MEANING of text. Similar meaning = nearby vectors, so "vacation" can match "paid time off" with no shared words.
- The model is a small transformer (6 layers) trained on ~1 billion related sentence pairs: pull related pairs together, push unrelated apart. That training happened before download.
- model.encode(): tokenize -> pass through layers (tokens look at each other for context) -> average into one 384-number vector.
- It does NOT generate text and learns nothing from your data.
- Use the SAME model for chunks and questions, otherwise vectors are not comparable.
- Similarity scores are relative rankings, not percentages. Look at the gap between 1st place and the rest. Bad chunking can flip the ranking (at size 150 an orphan chunk beat the right one).

### Types of embeddings
- By what is embedded: word (Word2Vec/GloVe, one fixed vector per word), sentence/passage (MiniLM, OpenAI, Cohere, BGE, E5; context-aware, used in RAG), image/audio/multimodal (CLIP), code.
- By construction: sparse keyword-based (TF-IDF, BM25; no AI, exact words only, misses synonyms), dense neural (ours; catches meaning, can miss rare exact terms), hybrid (both combined).
- Embeddings DO change with the model: vector size (384/768/1024/1536/3072), the vector space itself, quality, input limit (affects chunk size), language/domain.
- Rules: embed chunks and questions with the SAME model; switching models means re-embedding everything and rebuilding the index; compare models on the MTEB benchmark.

### Where AI is used in a RAG pipeline
- Loading: only for OCR (scanned PDFs/images) or speech-to-text; plain text needs none
- Chunking: usually rules only; semantic chunking uses an embedding model
- Embedding: YES (embedding model)
- Vector search (FAISS): no AI, just distance math
- Query rewriting (optional): LLM rewrites vague questions
- Reranking (optional): a reranker model re-orders the top ~20 candidates more accurately
- Prompt building: no AI, string formatting
- Generation: YES (LLM writes the answer)
- Answer checking (optional): a model verifies the answer is supported by the context
- Our tutorial = 2 AI models: all-MiniLM-L6-v2 (finds) and flan-t5-small (writes). Debug tip: if retrieved context is right but the answer is wrong, blame the generator; if the context is wrong, look at chunking/embedding.

## 5. FAISS (Step 4)
- Stores vectors and finds nearest ones fast; the "database" of RAG.
- Needs float32 vectors; needs the dimension (384) up front.
- IndexFlatL2: exact search using L2 (straight-line) distance. SMALLER distance = MORE similar (opposite of similarity scores). "Flat" = checks every vector; fine for small data, big systems use approximate indexes.
- FAISS stores only vectors, not text. Vector at position i corresponds to chunks[i].
- index.search(query_vectors, k) returns (distances, ids); query must be 2-D, so pass [question].
- Cosine similarity vs L2 distance: MiniLM outputs unit-length vectors, so FAISS's (squared) L2 distance = 2 - 2*cosine. High cosine <-> low distance, same ranking. Verified: cosine 0.390 -> distance 1.22; cosine 0.334 -> distance 1.33.

### Vector stores (general)
- A vector store keeps vectors + original text + metadata, and finds the nearest vectors to a query vector.
- Distance measures: L2, cosine, dot product (L2 and cosine rank the same for unit-length vectors).
- Exact (flat) search checks every vector; approximate (ANN) search is faster but may miss the true nearest. ANN methods: HNSW (graph, most popular), IVF (clusters), quantization/PQ (compression).
- FAISS is a vector SEARCH LIBRARY, not a full database: in-memory, no text/metadata storage, no saving by default, no server.
- Other types: libraries (Annoy, hnswlib); lightweight local stores (Chroma, LanceDB); dedicated vector DBs (Qdrant, Weaviate, Milvus, Pinecone); add-ons to existing DBs (pgvector, Elasticsearch/OpenSearch, MongoDB Atlas, Redis). Written from general knowledge; verify features/pricing before choosing.
- Swapping stores changes the add/search code only; the RAG idea is the same.

## Next
Step 5: retrieve -> augment (prompt template) -> generate with flan-t5-small. Possible issue: transformers 5.x may not support the "text2text-generation" pipeline name.
