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

## Step 5 - Retrieve, augment, generate (done)
- Added `retrieve(question, k=2)`, `augment(question, chunks)`, `generate(prompt)`, `ask(question, k=2)` to rag.py.
- Generator: google/flan-t5-small loaded with AutoTokenizer + AutoModelForSeq2SeqLM and `.generate()` (not the `pipeline("text2text-generation")` call, which may not work on transformers 5.x; the pipeline was not tried).
- Run at chunk_size=190, k=2, Q: "How many vacation days do I get?"
  - Retrieved: 'Paid Time Off Policy\nFull-time employees' (correct, first), 'Work From Home Policy\nOur company follow' (second)
  - Answer: `20`
- Result: the full pipeline works end to end. Correct answer, taken from the right chunk. Screenshot: screenshots/step5_rag_answer_vacation.png
- Observation: the answer is just "20", with no unit ("days"). Correct but terse; flan-t5-small tends to give very short answers.

### Step 5 - more test questions (chunk_size=190, k=2)

| Question | Retrieved (1st, 2nd) | Answer | Verdict |
|---|---|---|---|
| Which days must I be in the office? | Work From Home, Paid Time Off | Tuesday, Wednesday, and Thursday | Correct |
| What programming language do we use? | Tech Stack, Company Policy Manual | Python | Correct but incomplete (the doc also lists React and React Native) |
| Does PTO accrue monthly? | Paid Time Off, Company Policy Manual | yes | Correct |
| What is the dress code? | Company Policy Manual, Work From Home | a hybrid work model | WRONG: the document has no dress code |

Screenshots: screenshots/step5_test_office_days.png, step5_test_programming_language.png, step5_test_pto_monthly.png, step5_test_dress_code_wrong.png

Findings:
- Retrieval picked the right chunk first for all three answerable questions. The retrieval side is working.
- Dress code: the document does not cover it, but the model still answered, with "a hybrid work model" taken from the Work From Home chunk. It did not say "I don't know".
- Why: (1) top-k always returns k chunks, however irrelevant, so the generator is always handed something. Here the chunks were 'Company Policy Manual' and 'Work From Home'. (2) The prompt only says "using only the context", with no instruction for what to do when the answer is missing. (3) flan-t5-small is tiny and tends to pick out some phrase from the context instead of refusing.
- The second retrieved chunk is often filler ('Company Policy Manual'), because k=2 but only one chunk is relevant. This is harmless when the first chunk is right.
- Ideas to fix the dress-code case (NOT tried yet): a similarity threshold so low-scoring chunks are dropped and the system says "I don't know" without calling the model; a prompt line like "If the answer is not in the context, say I don't know"; a bigger generator.

## Next
- Try fixing the dress-code case (prompt change and/or similarity threshold) and re-run all four questions.
- Re-run the vacation question at chunk_size=150 and compare the answer.
- Small hit-rate test across chunk sizes
