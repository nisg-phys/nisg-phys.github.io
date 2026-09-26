---
layout: page
title: Job Search Agent — Pulsar (INSPIRE-HEP)
description: Natural-language search over INSPIRE-HEP physics job postings, with LLM query parsing, a pgvector semantic cache, multi-provider LLM failover, a guarded web-search fallback, and paper enrichment, deployed on Google Cloud Run with a Firebase-hosted frontend.
importance: 1
category: work
github: https://github.com/nisg-phys/INSPIRE-JOB-AGENT
---

[GitHub](https://github.com/nisg-phys/INSPIRE-JOB-AGENT) &middot; [Live Demo](https://pulsar-jobs-agent.web.app)

- Built a FastAPI service for natural-language search over INSPIRE-HEP physics job postings, where an LLM parses each query into validated (Pydantic) filters, asks a clarifying question when a query is ambiguous, and refuses off-topic requests, failing closed on unreadable output.
- Added a semantic cache on Postgres/pgvector with OpenAI embeddings, and an LLM router that fails over across Groq, Gemini, and OpenAI with per-provider cooldowns so a single provider outage does not fail a search.
- Designed a web-search fallback (Tavily) for queries INSPIRE has no postings for, with the LLM verifying each hit is a genuine matching posting; hardened against prompt injection by sanitizing and fencing page text as data and always linking the search-returned URL, never model output.
- Built a decoupled enrichment worker, scheduled daily on GitHub Actions, that attaches recent papers to each posting by INSPIRE institution ID; backed by 100+ pytest tests.
- Deployed on Google Cloud Run through a CI/CD pipeline using Workload Identity Federation, with a Firebase Hosting frontend; added Opik LLM tracing, structured logs, and a Grafana dashboard over Cloud Run and log-based metrics.

**Tech:** Python, FastAPI, PostgreSQL, pgvector, SQLAlchemy, Alembic, Groq, Gemini, OpenAI, Tavily, Opik, Docker, GitHub Actions, Google Cloud Run, Firebase Hosting, Grafana
