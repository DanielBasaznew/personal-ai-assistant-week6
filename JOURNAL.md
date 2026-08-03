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