import faiss
import numpy as np
import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

#------------------
#1. DOCUMENT INGESTION
#------------------
reader = PdfReader("document.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

#------------------
#2. CHUNKING
#------------------
chunk_size = 500
overlap = 50
chunks = []
st = 0
while st < len(text):
    end = st + chunk_size
    chunk = text[st:end]
    chunks.append(chunk)
    st = end - overlap

#------------------
#3. EMBEDDING
#------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)

#------------------
#4. Vector Store(FAISS)
#------------------
embeddings = np.array(embeddings).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

#------------------
#5. Retrieval
#------------------
query = input("Ask a Question: ")
query_embedding = embedding_model.encode([query])
query_embedding = np.array(query_embedding).astype("float32")
k = 3
relevant_chunks = []
distances, indices = index.search(query_embedding, k)
for i in indices[0]:
    relevant_chunks.append(chunks[i])

#------------------
#6. GENERATION(Using LLM)
#------------------
context = "\n\n".join(relevant_chunks)
prompt = f"""
Answer the question using the provided context.
Context:
{context}

Question:
{query}

If the context contains relevant information, use it to answer.
Do not use outside knowledge.

If the context does not contain enough information to answer, say:
"I could not find enough information in the provided document."
"""

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)
answer = response.choices[0].message.content
print("\nAnswer:")
print(answer)

