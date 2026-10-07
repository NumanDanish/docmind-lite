# 📄 DocMind Lite

Ask questions about any PDF and get answers with page numbers, powered by RAG
(Retrieval-Augmented Generation) and Google Gemini.

🔗 **Live demo:** https://docmind-lite-numan.streamlit.app

![DocMind Lite screenshot](assets/screenshot.png)

## ✨ Features

- Upload any text-based PDF and ask questions in plain language
- Answers cite page numbers, e.g. _(page 12)_
- "Sources" panel shows the exact text each answer came from
- Says "I couldn't find this in the document" instead of guessing
- Handles API rate limits automatically with retry
- Progress bar while the document is processed

## 🧠 How it works

1. **Read:** extracts text from the PDF page by page
2. **Chunk:** splits the text into overlapping pieces (800 characters, 150 overlap)
3. **Embed:** turns each piece into a vector with Gemini embeddings
4. **Retrieve:** finds the 5 pieces closest in meaning to the question (cosine similarity)
5. **Generate:** Gemini answers using only those pieces, with page citations

## 🛠️ Tech stack

Python · Streamlit · Google Gemini API (embeddings + generation) · pypdf

## 📁 Project structure

- `rag_engine.py`: the RAG logic (chunking, embedding, search, answering)
- `app.py`: the Streamlit web interface

## ▶️ Run locally

1. Clone the repo and install: `pip install -r requirements.txt`
2. Create `.streamlit/secrets.toml` with: `GEMINI_API_KEY = "your-key"`
3. Run: `streamlit run app.py`

## ⚠️ Limitations

- Text-based PDFs only (scanned PDFs need OCR)
- First 60 pages per document (to stay within free API limits)
- Broad questions like "summarize everything" only see part of the document
- Each question is answered independently (no follow-up memory yet)

## 🚀 Future improvements

- OCR support for scanned PDFs
- Hybrid search (meaning + keywords) and reranking
- Follow-up questions with conversation memory
- Accuracy evaluation with a test question set

## 👤 Author

**Mohammad Numan Danish** · [GitHub](https://github.com/NumanDanish)
