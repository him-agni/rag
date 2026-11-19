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

## Step 6 - Threshold + prompt fix, calibration run (done)
- Changed retrieve() to also return FAISS distances, added an "If the answer is not in the context, reply exactly: I don't know." line to the prompt, and made ask() drop chunks with distance > MAX_DISTANCE (if none are left, print "I don't know" without calling the model).
- This run used MAX_DISTANCE = 99 (nothing blocked) to measure distances.

| Question | Distances (best, 2nd) | Answer |
|---|---|---|
| How many vacation days do I get? | 1.221, 1.333 | 20 |
| Which days must I be in office? | 0.925, 1.553 | Tuesday, Wednesday, and Thursday |
| What programming language do we use? | 1.178, 1.749 | Python |
| Does PTO accrue monthly? | 0.783, 1.475 | yes |
| What is the dress code? | 1.553, 1.711 | a hybrid work model (still wrong) |

Screenshots: screenshots/step6_calib_*.png (vacation, office_days, programming_language, pto_monthly, dress_code)

Findings:
- The new prompt line did NOT change the dress-code answer: flan-t5-small ignored the "I don't know" instruction. The prompt fix is not reliable on this model.
- The threshold was not active in this run (99), so the dress-code result is expected.
- Distances separate the cases: best distance for answerable questions is 0.783 to 1.221 (worst = vacation, 1.221). Dress code (unanswerable) is 1.553. Gap = 1.221 to 1.553.
- Candidate threshold: MAX_DISTANCE = 1.4 (between the two). Paper check: keeps both vacation chunks (1.221, 1.333), keeps the best chunk for office/language/PTO and drops their filler second chunks (1.553, 1.749, 1.475), and drops both dress-code chunks (1.553, 1.711) -> "I don't know". NOT run yet.
- Caveat: only 4 answerable questions and 1 unanswerable, so 1.4 is tuned to these. Other out-of-scope questions could land below it.

## Step 6 - Threshold MAX_DISTANCE = 1.4 (unanswerable questions: works)

| Question | Distances (best, 2nd) | Answer |
|---|---|---|
| What is the dress code? | 1.553, 1.711 | I don't know (no relevant chunk found) |
| What is the CEO's name? | 1.504, 1.812 | I don't know (no relevant chunk found) |
| Is there free lunch? | 1.514, 1.712 | I don't know (no relevant chunk found) |

Screenshots: screenshots/step6_threshold1.4_dress_code.png, step6_threshold1.4_ceo_name.png, step6_threshold1.4_free_lunch.png

Findings:
- The dress-code question is fixed: it now gets "I don't know" instead of "a hybrid work model". The model is not called at all when no chunk passes the threshold.
- Two new unanswerable questions (CEO, free lunch) were also blocked. All three unanswerable best distances (1.504 to 1.553) are above 1.4.
- Margins at 1.4: answerable worst 1.221 is 0.18 below the threshold; unanswerable closest 1.504 is 0.10 above it. The upper side is the thinner one.
- Not yet re-run with the threshold on: the four answerable questions. By the calibration distances they should still pass, but that is unconfirmed.
- Code state: rag.py now holds the full pipeline (load -> chunk at 190/20 -> embed -> FAISS -> retrieve with distances -> threshold MAX_DISTANCE = 1.4 -> augment -> generate). The file ends with a single `ask("Is there free lunch?")` call. Committed and pushed to GitHub with this step.

## Next
- Re-run the four answerable questions with MAX_DISTANCE = 1.4 and confirm the answers are unchanged.
- Try a trickier unanswerable question that shares words with the document (e.g. "How many sick days do I get?") to test the edge of the threshold.
- Re-run the vacation question at chunk_size=150 and compare.
- Re-run the vacation question at chunk_size=150 and compare the answer.
- Small hit-rate test across chunk sizes
