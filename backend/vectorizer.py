import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class PDFQAEngine:
    def __init__(self, text, chunk_size=100, cache_file="qa_data.pkl"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.cache_file = cache_file

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
        words = text.split()
        return [' '.join(words[i:i+size]) for i in range(0, len(words), size)]

    def query(self, question, top_k=1):
        q_vec = self.model.encode([question])
        D, I = self.index.search(np.array(q_vec), top_k)
        return [self.chunks[i] for i in I[0]]

