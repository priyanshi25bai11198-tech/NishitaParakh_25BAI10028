from pypdf import PdfReader
reader = PdfReader("document.pdf")
text = ""
for pages in reader.pages:
    text += pages.extract_text()

print (text)