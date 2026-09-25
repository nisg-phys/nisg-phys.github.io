---
layout: page
title: Production RAG System — RAG-for-Python 3.14
description: A production RAG service combining dense vector search (Pinecone) and BM25 keyword scoring via Reciprocal Rank Fusion, with a change-aware ingestion pipeline, deployed on Google Cloud Run with a Firebase-hosted frontend.
importance: 2
category: work
github: https://github.com/nisg-phys/RAG-for-Python3.14
---

[GitHub](https://github.com/nisg-phys/RAG-for-Python3.14) &middot; [Live Demo](https://ragbot-python.web.app)

- Built a production RAG service with FastAPI and LangChain, combining dense vector search (Pinecone) with BM25 keyword scoring through Reciprocal Rank Fusion.
- Designed a change-aware ingestion pipeline with deterministic SHA-256 chunk IDs that skips unchanged sources and prunes stale vectors on edits.
- Containerized the service with Docker and deployed it on Google Cloud Run, paired with a static frontend on Firebase Hosting.
- Added Opik tracing for observability and moved chunk persistence to Backblaze B2 (S3-compatible object storage).

**Tech:** Python, FastAPI, LangChain, Pinecone, Backblaze B2, OpenAI, Groq (LLaMA 3.1), Opik, Docker, Google Cloud Run, Firebase Hosting
