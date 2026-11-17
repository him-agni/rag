# First RAG - progress notes

Tutorial: https://amanxai.com/2025/10/21/build-your-first-rag-system-from-scratch/
All tools are free and run locally: sentence-transformers, faiss-cpu, langchain-text-splitters, transformers (flan-t5-small). No API keys.

## Step 0 - Setup (done)
- venv created, packages installed. transformers 5.20.2 (newer than the tutorial; the text2text-generation pipeline in Step 5 may need a workaround).
- Import check printed `all good`.

## Step 1 - Load data (done)
- `my_knowledge.txt` loaded: 421 characters.

## Step 2 - Chunking (done)
Splitter: RecursiveCharacterTextSplitter (rule-based, no AI model).

| chunk_size | overlap | chunks | Notes |
|---|---|---|---|
| 80  | 20 | 10 | sentences cut mid-thought |
| 150 | 20 | 6  | heading separated from content; orphan chunk "Friday are optional work-from-home days." |
| 250 | 20 | 2  | unrelated topics merged (PTO + Tech Stack) |
| 190 | 20 | 4  | one chunk per policy section (best for this file) |

Overlap check:
- 190: no overlap (nothing was cut, so nothing to bridge)
- 150: 'Friday are optional'
- 80: 'be in the office on', 'and Friday are', 'React Native for'
- 150 with overlap=0: no overlap
Takeaway: overlap only appears where text had to be cut, and is capped by chunk_overlap.

## Step 3 - Embeddings (done, run at chunk_size=150)
- Model: all-MiniLM-L6-v2, shape (6, 384).
- Q: "How many vacation days do I get?"
  - 0.394 'Friday are optional work-from-home days.'  <- wrong chunk, narrowly on top
  - 0.390 'Paid Time Off Policy ...'                  <- correct chunk, essentially tied
  - rest <= 0.338, Tech Stack 0.034
- Q: "What programming languages do we use?"
  - 0.411 'Tech Stack Policy ...' <- clear winner, all others <= 0.125
Takeaway: bad chunking (the orphan chunk) can make retrieval pick the wrong chunk.

Re-run at chunk_size=190 (shape (4, 384)), Q: "How many vacation days do I get?"
  - 0.390 'Paid Time Off Policy ...'  <- correct, now first
  - 0.334 'Work From Home Policy ...' <- second (also mentions "days")
  - 0.131 'Company Policy Manual', 0.034 'Tech Stack Policy ...'
Takeaway: better chunking fixed the ranking, but the margin is only ~0.056, so top_k=2 is a useful safety net.

## Step 4 - FAISS index (done)
- index.ntotal = 4
- Q: "How many vacation days do I get?" top 2: ids [2 1], distances [1.2207712 1.3325232]
  - id 2 = Paid Time Off (correct), id 1 = Work From Home
- Same ranking as the cosine scores in Step 3 (0.390, 0.334). Link: distance = 2 - 2*cosine (1.2208 = 2 - 2*0.390; 1.3325 = 2 - 2*0.334).

## Next
- Step 5: retrieve + augment + generate (flan-t5-small)
- Small hit-rate test across chunk sizes
