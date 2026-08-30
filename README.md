# RAG Pipeline
A simple **Retrieval-Augmented Generation (RAG)** system that answers questions using information retrieved from a PDF document.

## Overview
The pipeline consists of six main stages:
1. Document Ingestion
2. Text Chunking
3. Embedding Generation
4. Vector Storage
5. Similarity Retrieval
6. LLM-based Generation

## Architecture
```text
PDF Document
     |
     v
Document Ingestion (pypdf)
     |
     v
Chunking
500 characters / 50 overlap
     |
     v
Embedding Model
all-MiniLM-L6-v2
     |
     v
FAISS Vector Store
IndexFlatL2
     |
     v
User Query
     |
     v
Query Embedding
     |
     v
Similarity Search
Top 3 Chunks
     |
     v
Context + Query
     |
     v
Groq LLM
     |
     v
Final Answer
```

## Design Justifications
### 1. Chunk Size — 500 Characters
The document is divided into **500-character chunks** with a **50-character overlap**.

500 characters provides a reasonable amount of context while keeping each retrieved chunk focused. The overlap helps preserve information that may fall at the boundary between two chunks.

These values are practical starting points and can be adjusted depending on the document and retrieval performance.

### 2. Embedding Model — all-MiniLM-L6-v2
The project uses:

`sentence-transformers/all-MiniLM-L6-v2`

This model was chosen because it is lightweight, runs locally, and produces **384-dimensional embeddings** suitable for semantic similarity.
The same model is used for document chunks and user queries so that they can be compared in the same vector space.

### 3. Vector Store — FAISS
The project uses **FAISS (Facebook AI Similarity Search)** with `IndexFlatL2`.
FAISS was chosen because the project contains a relatively small number of document chunks. `IndexFlatL2` provides simple and exact similarity search without requiring an external vector database.

### 4. Retrieval — Top 3
For every user query, the system retrieves the **3 most similar document chunks**.
This provides relevant context to the LLM without passing the entire document.

### 5. LLM — Groq
The retrieved chunks and the user's question are passed to an LLM through the **Groq API**.
The LLM is instructed to answer using the retrieved context and to indicate when the document does not contain enough information.

## Technologies Used
- **Python**
- **pypdf** — PDF text extraction
- **Sentence Transformers / Hugging Face** — text embeddings
- **FAISS** — vector similarity search
- **NumPy** — numerical operations
- **Groq API** — LLM generation
- **python-dotenv** — environment variable management

## Project Structure
```text
SDC Track2-Task/
│
├── document.pdf
├── rag.py
├── requirements.txt
├── README.md
├── .gitignore
└── .env
```

> `.env` contains the Groq API key and is excluded from GitHub using `.gitignore`.

## Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory:
```text
GROQ_API_KEY=your_api_key_here
```

Do not commit the `.env` file to GitHub.
## Running the Project

Run:
```bash
python rag.py
```

The program will ask:

```text
Ask a Question:
```
Enter a question related to the PDF document.
The system retrieves relevant chunks and uses them as context to generate the final answer.

## Example
```text
Ask a Question: What are the different types of flip-flops?

Answer:
The document lists RS, T, D and JK flip-flops.
```
### Example Questions

Try questions that are directly related to the provided document:
- What are the different types of flip-flops?
- What are the different types of shift registers mentioned in the document?
- What number systems are mentioned in the document?
- What topics are included under combinational logic?
- What are the different types of codes mentioned in the document?

## RAG Workflow
```text
Document
   |
   v
Text Extraction
   |
   v
Chunking
   |
   v
Embeddings
   |
   v
FAISS
   |
   v
Query
   |
   v
Relevant Chunks
   |
   v
LLM
   |
   v
Final Answer
```

## Requirements
All required Python packages are listed in `requirements.txt`.
Install them using:
```bash
pip install -r requirements.txt
```
