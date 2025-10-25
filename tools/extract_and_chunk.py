import pdfplumber, json, re
from pathlib import Path
from tqdm import tqdm

def chunk_text(text, size=500, overlap=80):
    words = text.split()
    chunks = []
    for i in range(0, len(words), size - overlap):
        chunks.append(" ".join(words[i:i+size]))
    return chunks

def extract_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            for chunk_id, chunk in enumerate(chunk_text(text)):
                yield {
                    "source": path.name,
                    "page": i,
                    "chunk_id": chunk_id,
                    "text": chunk
                }

if __name__ == "__main__":
    docs = list(Path(".").glob("*.pdf"))
    all_chunks = []
    for doc in tqdm(docs):
        all_chunks.extend(list(extract_from_pdf(doc)))
    Path("chunks.jsonl").write_text("\n".join(json.dumps(c) for c in all_chunks))

