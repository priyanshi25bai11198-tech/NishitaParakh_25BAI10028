from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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
print("No. of embeddings: ", len(embeddings))
print("Embedding size: ", len(embeddings[0]))
