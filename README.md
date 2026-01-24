# FOMC RAG: Hybrid Extraction & Retrieval System

This repository contains the core architecture for a Retrieval-Augmented Generation (RAG) pipeline designed specifically for complex financial documents (FOMC minutes). 

This is the public release of the **base system**. While the internal project includes uncertainty quantification, red-teaming, and teacher-model grading, this repository focuses solely on the ingestion, hybrid extraction, and vector retrieval pipelines.

## The Problem
Standard OCR tools fail on FOMC documents because they rely heavily on dense tables and mixed-column layouts. To solve this, I built a hybrid extractor that routes simple text pages to high-speed OCR and complex table pages to a Vision Language Model (VLM).

## System Architecture

I used **Surya** for layout detection and **Qwen2-VL** for extracting data from economic projection tables. The retrieved context is fed into **Llama 3** for generation.

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
```
