#------------------
#DOCUMENT INGESTION
#------------------
from pypdf import PdfReader
reader = PdfReader("document.pdf")
text = ""
for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print(text)

#------------------
#CHUNKING
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

print("Number of Chunks: ", len(chunks))
print("first chunk: ", chunks[0])


#------------------
#CHUNKING
#------------------