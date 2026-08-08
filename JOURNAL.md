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

---

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

---

# Week 6, Day 3: Metadata Filtering + Self-Querying Router

> **Core Takeaway:** Hardcoding filter logic creates fragile search systems. By prompting Gemini 3.1 Flash-Lite with strict Pydantic JSON schema constraints, I implemented a "Self-Querying" intent router that dynamically extracts source filters, document types, and cleaned query strings before passing parameters to hybrid retrieval.

## 1. Query Router Model Evaluation
I evaluated the LLM intent router across several distinct query patterns to verify extraction accuracy:

* **Source-Specific Queries:** `"What does resnet_paper say about residual learning?"`
  * **Parsed Intent:** `clean_query="residual learning"`, `source_filter="resnet_paper"`, `doc_type="paper"`
  * **Retrieval Effect:** Pre-filtered both BM25 and ChromaDB indexes, restricting search space directly to target embeddings.
* **Document-Type Queries:** `"In my notes, what did I write about RAG?"`
  * **Parsed Intent:** `clean_query="what did I write about RAG"`, `source_filter="my_notes"`, `doc_type="notes"`
  * **Retrieval Effect:** Restricted vector and keyword search space strictly to note chunks.
* **Explicit Global Queries:** `"Search everything for Frankenstein"`
  * **Parsed Intent:** `clean_query="Frankenstein"`, `source_filter=None`, `doc_type=None`
  * **Retrieval Effect:** Overrode source filters to perform full-breadth hybrid search across the entire database.
* **Conversational Inputs:** `"Hello, how are you today?"`
  * **Parsed Intent:** `requires_documents=False`
  * **Retrieval Effect:** Bypassed vector and keyword retrieval entirely to eliminate unnecessary inference calls.

## 2. Key Insights
1. **Clean Query Extraction:** Stripping operational phrasing (e.g., *"search in my notes for"*) improves keyword matching scores in BM25 because filler tokens aren't competing for document frequency weight.
2. **Dual-Index Synchronization:** Writing rich metadata (`document_name`, `document_type`, `tags`, `ingested_at`) to both ChromaDB and BM25 simultaneously ensures filter clauses work identically across vector and keyword retrieval.

---

# Week 6, Day 4: Persistent Semantic & Episodic Memory Layer

> **Core Takeaway:** To give an AI assistant long-term memory, we distinguish between Episodic Memory (raw timestamped conversation logs) and Semantic Memory (distilled facts). I built an LLM-driven extraction pipeline using Gemini 3.1 Flash-lite and SQLite `UPSERT` logic to manage long-term state across sessions.

## 1. Memory Extractor Prompt Engineering & Edge-Case Evaluation
I designed a system prompt and Pydantic output schema to evaluate how reliably Gemini 3.1 Flash-lite extracts personal state updates across 10 stress-test queries:

* **Successes:** The model isolated multiple distinct entities in a single sentence (location + allergies), captured negative preferences ("dislikes horror movies"), and successfully ignored conversational noise and general technical prompts.
* **Failure - Temporary States:** The model over-extracted temporary user conditions (e.g., storing "exhausted due to lack of sleep"). I identified that semantic extraction prompts require explicit temporal guidelines to prevent temporary states from polluting long-term memory.
* **Failure - Third-Party Bleed:** Despite instructions restricting extraction to user facts, the model extracted third-party data (`brother_occupation`).
* **Failure - Key Inconsistency:** During an update test, the LLM saved the user's location under `key='location'`, but later attempted an update using `key='current_city'`. In exact-key key-value stores, this fails to trigger SQLite's `ON CONFLICT DO UPDATE` clause, resulting in duplicate records.

## 2. Production System Gap (Mem0)
Compared to exact-key lookups, production memory frameworks like Mem0 solve the "Key Inconsistency" problem by storing facts as vector embeddings. Instead of relying on the LLM to guess exact string keys (`location` vs `current_city`), they use semantic vector search to identify related prior facts, followed by an LLM consolidation pass to handle updates.

---

# Week 6, Day 5: Personal AI Assistant - Full Integration & Stress Testing

> **Core Takeaway:** I unified an LLM Query Router, Hybrid RRF Search, and an Automatic Memory Extraction Engine into a single, production-style personal assistant. The architecture dynamically conditions LLM generation on retrieved document chunks and persistent user facts.

## 1. Multi-Model Pipeline & System Integration
I conducted an end-to-end evaluation to verify real-time interactions across all AI modules:
* **UI vs. LLM Command Routing:** Diagnosed an edge case where terminal parsing (`if input.startswith("forget ")`) intercepted user messages before reaching the LLM memory extractor. I resolved this by isolating CLI admin commands (`/forget <key>`) from natural language statements, allowing the Gemini memory extraction engine to handle natural conversation seamlessly.
* **In-Context Prompt Modulation:** Verified that extracted semantic memory directly modulates LLM output. When asking the model to explain residual learning from the ResNet paper, the system retrieved paper chunks via Hybrid Search AND applied a stored user preference (`clear, step-by-step explanations`) to structure the response.
* **Dynamic Intent Routing:** Confirmed that the self-query router reliably switches between targeted paper lookups, global database searches, and standard conversational inference without hallucinating filter constraints.

## 2. Production Scaling Gap (Mem0 & Copilot)
While our assistant effectively implements the core architecture of systems like Microsoft Copilot and Rewind AI (semantic extraction -> persistent storage -> prompt injection), enterprise implementations scale using two key techniques:
1. **Vector-Based Fact Retrieval:** Instead of injecting all stored facts into every system prompt, production systems embed facts and perform vector similarity search to inject only the top $k$ relevant facts per conversation turn.
2. **LLM-Driven Conflict Resolution:** Replacing exact-key `UPSERT` operations with secondary LLM consolidation passes to maintain historical timelines (e.g., tracking movement from city A to city B over time).