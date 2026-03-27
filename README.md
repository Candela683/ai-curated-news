

# 🧠 Cluster Summarization Pipeline

## What this project does

This project processes large volumes of news data and turns them into structured topic summaries.
The system combines semantic clustering and LLM summarization to convert continuous news streams into structured topic-level insights, with a focus on identifying and tracking trending topics in near real-time.

including:

* collects news articles (GDELT RSS)
* groups similar articles into clusters based on meaning
* generates a concise summary for each cluster using an LLM
* provides an API to retrieve recent summaries

The output is a stream of organized, human-readable topic summaries instead of raw articles.

---

## How it works

### 1. Embedding and clustering

* Each news title is converted into a vector using a sentence transformer
* New items are compared against existing clusters:

  * if similarity is high → assigned to that cluster
  * otherwise → grouped into new clusters using HDBSCAN
* Cluster centroids and metadata are updated incrementally

Core logic: 

---

### 2. LLM summarization

* When a cluster accumulates enough new items:

  * recent titles are collected
  * formatted into a prompt
  * sent to an LLM for summarization

* The summary is stored in SQLite and optionally exported as JSONL

Core logic: 

---

### 3. Pipeline execution

* A runner script executes the full pipeline in sequence:

  1. data ingestion
  2. clustering
  3. summarization

* Runs continuously at a fixed interval (default: 15 minutes)

Core logic: 

---

### 4. API service

* A FastAPI service exposes recent cluster summaries
* Supports pagination using cursor-based queries

Example:

```
GET /clusters/recent
```

Core logic: 

---

## Output

Each cluster includes:

* cluster_id
* summary text
* related URLs
* timestamp for ordering

---

## Summary

The system combines semantic clustering and LLM summarization to convert continuous news streams into structured topic-level insights.
