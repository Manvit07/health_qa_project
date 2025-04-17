import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import re


class PDFQAEngine:
    def __init__(self, text, chunk_size=500, cache_file="qa_data.pkl"):
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

        helpful_keywords = ['treatment', 'care', 'doctor', 'medicine', 'medication', 'rest', 'hydration', 'symptoms', 'pain relief', 'consult', 'reduce', 'fever']

        # Try to find a helpful chunk first
        for chunk in retrieved:
            if any(keyword in chunk.lower() for keyword in helpful_keywords):
                clean_text = self._remove_unwanted_sections(chunk)
                important_lines = self._extract_action_sentences(clean_text)
                final_answer = ' '.join(important_lines)
                if final_answer:
                    return [final_answer]

        # If no helpful chunk found, just use the first and clean it
        clean_text = self._remove_unwanted_sections(retrieved[0])
        important_lines = self._extract_action_sentences(clean_text)
        final_answer = ' '.join(important_lines)
        if not final_answer:
            final_answer = clean_text
        return [final_answer]

    def _remove_unwanted_sections(self, text):
        # Remove sections starting from "KEY TERMS" onward
        if "KEY TERMS" in text:
            text = text.split("KEY TERMS")[0]
        # Remove extra definitions manually
        text = re.sub(r"\b[A-Z][a-z]+ —.*?(\.|\n)", "", text)  # Removes "Erythrocytes — definition."
        return text.strip()
    
    def _extract_action_sentences(self, text):
        # Keep sentences that have action keywords
        action_keywords = ["treat", "treatment", "care", "rest", "drink", "hydration", "doctor", "pain", "medication", "reduce", "seek medical", "consult"]
        sentences = re.split(r'(?<=[.!?])\s+', text)  # split into sentences
        important = [s for s in sentences if any(word in s.lower() for word in action_keywords)]
        return important