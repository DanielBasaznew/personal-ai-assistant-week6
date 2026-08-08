# Week 6, Day 1: BM25 vs. Semantic Search Comparison

> **Core Takeaway:** Pure semantic search (embeddings) and pure keyword search (BM25) fail in completely different ways. BM25 is unbeatable for exact strings, unique proper nouns, and numeric hyperparameters, but it has zero concept of meaning. Semantic search understands intent and synonyms, but can miss exact jargon or acronyms.

## 1. Where BM25 Clearly Outperformed Semantic Search
BM25 dominated when queries contained exact technical terms, specific numbers, or unique proper nouns that existed verbatim in the corpus:

* **`"Adam optimizer beta1 0.9"`**
  * **Why BM25 won:** It immediately surfaced the exact Transformer paper sentence (`with β1 = 0.9, β2 = 0.98 and ϵ = 10−9`) with a massive score of **27.9389**.
* **`"BLEU scores English-to-German"`**
  * **Why BM25 won:** It locked directly onto Table 2 from the research paper comparing exact WMT translation metrics (Score: **23.4029**).
* **`"Modern Prometheus Wollstonecraft"`**
  * **Why BM25 won:** Rare proper nouns are BM25's superpower; it instantly identified the *Frankenstein* title page block (Score: **21.6110**).

## 2. Where Semantic Search Clearly Outperformed BM25
Semantic search won whenever queries used paraphrased concepts, broad questions, or words with multiple meanings:

* **`"Why did the creator feel guilty about his monster?"`**
  * **Why Semantic won:** It successfully retrieved dialogue addressing Frankenstein's crimes and relationship with the creature (`Dist: 0.4264`), whereas BM25 got distracted by irrelevant sentences that happened to repeat the word `"circumstance"`.
* **`"How to build custom machine learning pipelines?"`** *(The "Keyword Trap" Test)*
  * **Why Semantic won:** BM25 blindly matched the word `"custom"` to a quote about *"the custom of the Irish"* in Frankenstein. Semantic search ignored the literal word match and correctly retrieved the TensorFlow/tensor2tensor ML training models from the research paper (`Dist: 0.6835`).
* **`"How does the model calculate attention between words?"`**
  * **Why Semantic won:** It retrieved conceptual explanations of masking and self-attention layers without requiring exact token overlap for the word `"calculate"`.

# Week 6, Day 2: Hybrid Search via Reciprocal Rank Fusion (RRF)

> **Core Takeaway:** Reciprocal Rank Fusion (RRF) solves the score scale incompatibility between BM25 and vector embeddings by ranking documents purely by position ($1 / (\text{rank} + k)$). Combining a $top\_k \times 3$ candidate pool from both search methods guarantees that exact matches and conceptual matches both float to the top.

## 1. Hybrid Search Validation Results
Running our 6 validation queries through `HybridSearchEngine` yielded a **100% success rate** across both technical and conceptual query types:

* **Exact Technical/Parametric Queries:**
  * `"Adam optimizer beta1 0.9"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01639`]
  * `"BLEU scores English-to-German"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01639`]
  * `"Modern Prometheus Wollstonecraft"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01639`]
* **Broad Conceptual Queries:**
  * `"How does the model calculate attention between words?"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01600`]
  * `"Why did the creator feel guilty about his monster?"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01600`]
  * `"What training hardware and GPUs were used?"` $\rightarrow$ **Rank 1 (`BM25 + Semantic`)** [RRF: `0.01626`]

## 2. Key Observations
1. **Agreement Boost:** When both BM25 and Semantic Search identify the same document in their candidate pools, the combined RRF score pushes it straight to Rank 1.
2. **Fallback Safety:** If Semantic Search misses a rare proper noun (like `Wollstonecraft`), BM25 still captures it and retains it in the final results list without crashing or getting buried.

# Week 6, Day 3: Metadata Filtering + Self-Querying Router

> **Core Takeaway:** Hardcoding filter logic creates fragile search systems. Using an LLM (Gemini 2.5 Flash) to parse user intent into a structured `QueryIntent` object enables "Self-Querying" — automatically extracting source filters, document types, and cleaned query strings before passing them to hybrid retrieval.

## 1. Query Router Evaluation
We evaluated `parse_query_intent` across several distinct query patterns:

* **Source-Specific Queries:** `"What does resnet_paper say about residual learning?"`
  * **Parsed Intent:** `clean_query="residual learning"`, `source_filter="resnet_paper"`, `doc_type="paper"`
  * **Retrieval Effect:** Applied pre-filtering in both BM25 and ChromaDB, narrowing search space directly to the target paper.
* **Document-Type Queries:** `"In my notes, what did I write about RAG?"`
  * **Parsed Intent:** `clean_query="what did I write about RAG"`, `source_filter="my_notes"`, `doc_type="notes"`
  * **Retrieval Effect:** Restricted search strictly to notes chunks.
* **Explicit Global Queries:** `"Search everything for Frankenstein"`
  * **Parsed Intent:** `clean_query="Frankenstein"`, `source_filter=None`, `doc_type=None`
  * **Retrieval Effect:** Overrode all source filters to perform full-breadth hybrid search across the whole database.
* **Conversational Inputs:** `"Hello, how are you today?"`
  * **Parsed Intent:** `requires_documents=False`
  * **Retrieval Effect:** Bypasses document retrieval entirely to save compute.

## 2. Key Insights
1. **Clean Query Extraction:** Stripping operational phrasing (e.g., *"search in my notes for"*) improves keyword matching scores in BM25 because filler tokens aren't competing for document frequency weight.
2. **Dual-Index Synchronization:** Writing rich metadata (`document_name`, `document_type`, `tags`, `ingested_at`) to both ChromaDB and BM25 simultaneously ensures filter clauses work identically across vector and keyword retrieval.

# Week 6, Day 4: Persistent Semantic & Episodic Memory Layer

> **Core Takeaway:** To give an AI assistant long-term memory, we distinguish between Episodic Memory (raw timestamped conversation logs) and Semantic Memory (distilled facts). A simple SQLite database with `UPSERT` handles basic updates, but LLM extraction requires careful tuning to prevent storing temporary states or third-party facts.

## 1. Memory Extractor Edge-Case Analysis
Running 10 stress-test queries through our Gemini 2.5 Flash extractor revealed distinct strengths and weaknesses in our naive extraction prompt:

* **Successes:** The extractor perfectly isolated multiple facts in a single sentence (location + allergies), accurately captured negative preferences ("dislikes horror movies"), and successfully ignored conversational noise + technical queries.
* **Failure - Temporary States:** The model eagerly stored "exhausted due to lack of sleep." Semantic memory should only store persistent facts; temporary states pollute long-term context.
* **Failure - Third-Party Bleed:** Despite instructions to only store facts about the user, it stored a fact about the user's brother (`brother_occupation`). 
* **Failure - Key Inconsistency:** During an update test, the LLM initially saved the user's city under `key='location'`, but later tried to update it using `key='current_city'`. In a pure SQLite key-value store, this fails to trigger the `ON CONFLICT DO UPDATE` clause, resulting in duplicate, conflicting records. 

## 2. Production System Gap (Mem0)
Compared to our exact-key lookup system, production memory agents like Mem0 solve the "Key Inconsistency" problem by storing facts as vector embeddings. Instead of relying on the LLM to guess the exact string key (`location` vs `current_city`), they use semantic similarity to find related past facts, and then use a secondary LLM pass to consolidate contradictory timelines.