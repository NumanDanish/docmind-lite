import math
import time
from google import genai
from google.genai import types
from pypdf import PdfReader

MODEL = "gemini-3.5-flash-lite"
EMBED_MODEL = "gemini-embedding-001"
MAX_PAGES = 60   # keeps us inside the free limits

SYSTEM_PROMPT = """You answer questions using ONLY the document excerpts provided.
Explain in simple language.
After each fact, cite the page number like this: (page 12).
If asked to summarize, summarize what the excerpts contain and say it covers only part of the document.
If the excerpts do not contain the answer, say exactly:
"I couldn't find this in the document."
Never use outside knowledge and never guess."""


def get_client(api_key):
    return genai.Client(api_key=api_key)


def load_chunks(file, chunk_size=800, overlap=150):
    reader = PdfReader(file)
    total_pages = len(reader.pages)
    chunks = []
    for page_number, page in enumerate(reader.pages[:MAX_PAGES], start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())
        start = 0
        while start < len(text):
            chunks.append({"page": page_number, "text": text[start:start + chunk_size]})
            start += chunk_size - overlap
    return chunks, total_pages


def embed(client, texts, task):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task),
    )
    return [e.values for e in result.embeddings]


def embed_all(client, texts, on_progress=None, batch_size=50):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        while True:
            try:
                vectors += embed(client, batch, "RETRIEVAL_DOCUMENT")
                break
            except Exception as e:
                if "429" in str(e):
                    time.sleep(60)
                else:
                    raise
        if on_progress:
            on_progress(len(vectors), len(texts))
    return vectors


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    size_a = math.sqrt(sum(x * x for x in a))
    size_b = math.sqrt(sum(y * y for y in b))
    return dot / (size_a * size_b)


def search(client, question, chunks, vectors, top_k=5):
    query_vector = embed(client, [question], "RETRIEVAL_QUERY")[0]
    scores = [cosine_similarity(query_vector, v) for v in vectors]
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    return ranked[:top_k]


def answer(client, question, top):
    context = "\n\n".join(f"[Page {c['page']}] {c['text']}" for score, c in top)
    prompt = f"Document excerpts:\n{context}\n\nQuestion: {question}"
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
    )
    if response.text is None:
        return "Sorry, I couldn't generate an answer for this question. Please try rephrasing it."
    return response.text