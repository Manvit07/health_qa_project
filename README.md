# Medical Health Assistant Bot (PDF-Based QA System)

This project is a lightweight, offline **Medical Health Assistant** that helps users get simple, actionable medical advice from a medical handbook PDF.  
It uses **Semantic Search + Chunked Retrieval** to answer health-related queries like "What should I do for a fever?" without requiring internet APIs.

---

## 🚀 Features

- **PDF-Based Knowledge:** Load any medical handbook (Merck Manual, Red Cross Manual, etc.) and build a local intelligent assistant.
- **Semantic Understanding:** Uses `SentenceTransformer` models to understand user questions meaningfully.
- **Fast Search Engine:** Powered by `FAISS` for real-time answer retrieval.
- **Action-Focused Responses:** Filters and extracts practical "what to do" sentences.
- **Optimized for Normal Users:** Removes unnecessary medical jargon, technical terms, and complicated descriptions.
- **Lightweight Deployment:** No heavy language models (like GPT) needed — works even on laptops.
- **Customizable Dataset:** Easily switch PDFs to target different health domains (home care, emergencies, travel medicine, etc.)

---

## 🏛️ Tech Stack

| Tech | Purpose |
|:-----|:--------|
| SentenceTransformers (MiniLM) | Semantic embedding of questions and document chunks |
| FAISS | Fast similarity search over embedded vectors |
| PyMuPDF (fitz) | PDF text extraction |
| Huggingface Transformers (optional) | Summarization (if needed) |
| Python (3.9+) | Core backend language |

---

## 🗂️ Project Structure

| File/Folder | Purpose |
|:------------|:--------|
| `app.py` | Backend server (Flask/FastAPI app) to receive queries and return answers |
| `vectorizer.py` | Core class to chunk PDF, embed text, build FAISS index, and perform smart retrieval |
| `pdf_reader.py` | Extract raw text from PDF documents |
| `health_book.pdf` | The loaded medical handbook (current knowledge base) |
| `qa_data.pkl` | Pickle file storing FAISS index and embeddings for fast reload |
| `requirements.txt` | All Python dependencies listed for easy setup |

---

## 📦 Installation Guide

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/medical-health-assistant.git
cd medical-health-assistant
