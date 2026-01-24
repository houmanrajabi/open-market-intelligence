# FOMC RAG: Hybrid Extraction & Retrieval System

This repository contains the core architecture for a Retrieval-Augmented Generation (RAG) pipeline designed specifically for high-complexity financial documents (FOMC minutes).

This is the public release of the **base system**. While the internal project includes uncertainty quantification, red-teaming, and teacher-model grading, this repository focuses solely on the ingestion, hybrid extraction, and vector retrieval pipelines.

## The Problem
Standard OCR tools are fundamentally unsuited for financial reporting. They rely on linear text extraction, which destroys the semantic structure of "dot plots," economic projection tables, and dual-column layouts. When a standard RAG system attempts to process these documents, it flattens tables into nonsense strings, leading to hallucinations when the model attempts to interpret the numbers.

## System Architecture

To address this, I designed a pipeline that treats text and visual data as distinct streams. Rather than forcing all content through a single OCR engine, the system intelligently routes pages based on layout density, ensuring that tabular data preserves its row-column relationships before it ever reaches the embedding model.

### 1. Ingestion (The "Split" Strategy)
I utilize a page-level classifier to determine the optimal extraction method:
* **Text-Heavy Pages:** Processed via **Surya OCR** to accurately reconstruct column reading order.
* **Data-Heavy Pages:** Routed to **Qwen2-VL**, a Vision Language Model that transcribes complex tables directly into Markdown.
* **Reconstruction:** The streams are merged and injected with document-level metadata (e.g., Meeting Date) to prevent the temporal confusion common in financial RAG.

### 2. Retrieval (The "Filter")
Precision in finance is binary—it is either correct or it is a liability. Pure vector search is insufficient for specific targets (like "2.0%"). I implemented a hybrid approach:
* **Hybrid Search:** Queries are routed to both **ChromaDB** (semantic search) and a **BM25 Index** (keyword/entity search).
* **Reranking:** A **Cross-Encoder** re-scores the retrieved candidates, effectively filtering out "semantically similar but factually irrelevant" noise to present only the top 5 contexts.

### 3. Generation (Code-Driven Visualization)
Retrieving numbers is only half the battle; visualizing them is where LLMs often fail.
* **Llama 3 (70B):** Acts as the reasoning engine.
* **Code Execution:** Instead of asking the model to "draw" a chart (which leads to visual hallucinations), the model generates Python code. This code is executed in a sandbox to render mathematically accurate plots from the retrieved data.

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion [Ingestion Phase]
        A[Raw FOMC PDFs] --> B{Page Classifier}
        B -->|Text Layout| C[Surya OCR]
        B -->|Tables/Charts| D[Qwen2-VL Vision Model]
        
        C --> E[Merger & Layout Reconstructor]
        D --> E
        
        %% New Step: Metadata Injection
        E --> F[Metadata Injector]
        F -->|Enriched Text| G[Semantic Chunking]
        
        %% Storage: Hybrid Approach
        G --> H[(ChromaDB Vector Store)]
        G --> I[(BM25 Keyword Index)]
    end

    %% Retrieval Pipeline
    subgraph Retrieval [Retrieval Phase]
        Q[User Query] --> J{Query Router}
        J -->|Semantic Search| H
        J -->|Exact Terms| I
        
        H --> K[Retrieval Fusion]
        I --> K
        
        %% Critical Addition: Reranking
        K --> L[Cross-Encoder Reranker]
        L -->|Top 5 Contexts| M[Llama-3 70B Generator]
    end

    %% Generation & Output
    subgraph Output [Generation Phase]
        M -->|Text Response| N[Final Answer]
        
        %% Critical Addition: Code Interpreter for Graphs
        M -->|Python Code| O[Code Execution Sandbox]
        O --> P[Visualization Graph]
    end

    %% Styling
    style L fill:#f96,stroke:#333,stroke-width:2px,color:black
    style O fill:#f96,stroke:#333,stroke-width:2px,color:black
    style I fill:#ff9,stroke:#333,stroke-width:2px,color:black
