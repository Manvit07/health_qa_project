import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np

class PDFQAEngine:
    def __init__(self, text, chunk_size=500, cache_file="qa_data.pkl"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_file = cache_file
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

        if os.path.exists(cache_file):
            print("📦 Loading cached data...")
            with open(cache_file, 'rb') as f:
                self.chunks, self.embeddings = pickle.load(f)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(np.array(self.embeddings))
        else:
            print("⚙️ Building index from scratch...")
            self.chunks = self._chunk_text(text, chunk_size)
            self.embeddings = self.model.encode(self.chunks)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(np.array(self.embeddings))
            with open(cache_file, 'wb') as f:
                pickle.dump((self.chunks, self.embeddings), f)

    def _chunk_text(self, text, size):
        sentences = text.split('. ')
        chunks = []
        current_chunk = ''

        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < size:
                current_chunk += sentence + '. '
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + '. '

        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def query(self, question, top_k=5):
        q_vec = self.model.encode([question])
        D, I = self.index.search(np.array(q_vec), top_k)
        retrieved = [self.chunks[i] for i in I[0]]

        helpful_keywords = ['treatment', 'care', 'doctor', 'medicine', 'medication', 'rest', 'hydration', 'symptoms', 'pain relief']

        # Try to find a helpful chunk
        for chunk in retrieved:
            if any(keyword in chunk.lower() for keyword in helpful_keywords):
                # Summarize this helpful chunk
                simple_summary = self.summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                return [simple_summary]

        # If no helpful chunk found, just summarize the first one
        simple_summary = self.summarizer(retrieved[0], max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return [simple_summary]
