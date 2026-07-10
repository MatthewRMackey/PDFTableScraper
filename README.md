# ESG Document Intelligence Pipeline

**Matthew Mackey | 2023**

> An end-to-end pipeline for extracting and structuring tabular ESG data from corporate sustainability documents at scale, using GPT-3.5 for relevance classification and GPT-4 for structured data extraction.

---

## Overview

Corporate ESG (Environmental, Social, and Governance) reports contain critical sustainability metrics — emissions, water usage, waste, energy consumption — buried across hundreds of pages of narrative text, footnotes, and mixed-content tables. Manually extracting this data across thousands of company filings is infeasible.

This pipeline automates the full extraction workflow: identifying which pages in a document contain relevant tabular ESG data, discarding the rest, and passing only the relevant pages to a more capable model for structured extraction.

The two-stage architecture balances cost and accuracy: GPT-3.5 handles the cheap classification step at scale; GPT-4 handles precise table transcription on the already-filtered subset.

---

## Pipeline

```
[Raw PDFs]
    ↓
[PreProcessor] — GPT-3.5 classifies each page: True (contains ESG table) / False
    ↓
[Trimmed PDFs] — only relevant pages retained
    ↓
[TableProcessor] — GPT-4 transcribes table contents into structured CSV format
    ↓
[Structured ESG Data]
```

---

## Stage 1: PreProcessor

**Script:** `PreProcessor.py`

For each raw PDF:
1. Iterates through all pages using PyPDF2
2. Extracts page text and checks token count against the GPT-3.5 context limit (4096 tokens, 85% threshold)
3. Sends each page to GPT-3.5 with a classification system prompt — the model returns `True` if the page contains a relevant ESG data table, `False` otherwise
4. Pages marked `True` are retained; all others are discarded
5. A trimmed PDF containing only the relevant pages is written to disk
6. Processed filenames are tracked in a CSV to prevent reprocessing on subsequent runs
7. Rate limiting is applied between pages (3s) and between documents (10s) to avoid API throttling

---

## Stage 2: TableProcessor

**Script:** `TableProcessor.py`

For each trimmed PDF:
1. Iterates through the retained pages
2. Extracts page text and checks token count against the GPT-4 context limit
3. Sends each page to GPT-4 with a table transcription system prompt — the model outputs the table contents in a structured format
4. Structured outputs are written to text files per document
5. Processed filenames are tracked to prevent reprocessing

---

## Results

- **~97% accuracy** on tabular data extraction from well-formatted ESG pages
- Identified precision/recall limitations at production scale as the primary barrier to full deployment across thousands of company filings — some pages contain ambiguous or poorly formatted tables that the classifier struggles to handle consistently
- The cost differential between GPT-3.5 (classification) and GPT-4 (extraction) makes the two-stage approach economically viable at scale compared to running GPT-4 on all pages

---

## Technical Stack

- **Language:** Python
- **PDF processing:** PyPDF2 (evaluated Tesseract OCR; switched due to hallucination artifacts in OCR output)
- **LLMs:** GPT-3.5-turbo (classification), GPT-4 (extraction)
- **Tokenization:** tiktoken
- **Rate limiting:** built-in via time.sleep

---

## Limitations and Future Work

- Current implementation processes one document at a time sequentially; parallel processing would significantly improve throughput
- There is a fundamental tradeoff in intelligent document processing: high precision is achievable for consistently structured documents, but general methods applied to inconsistent structure will always yield lower precision. ESG reports vary enormously in layout across companies and years — this is a structural constraint on the problem, not a failure of implementation. The ~97% accuracy figure reflects performance on well-formatted pages; production deployment across thousands of filings would encounter the full distribution of document quality.
- Output format is currently free-text structured by GPT-4; a schema-enforced output format (JSON or CSV with defined column headers) would improve downstream usability
- A follow-on project (NuExtract3 document QA) addresses some of these limitations using PDF-to-markdown conversion with hybrid retrieval
