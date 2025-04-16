from flask import Flask, request, jsonify
from flask_cors import CORS
from pdf_reader import extract_text
from vectorizer import PDFQAEngine

app = Flask(__name__)
CORS(app)  # Allow frontend access

book_text = extract_text('backend/data/health_book.pdf')
engine = PDFQAEngine(book_text)

@app.route('/ask', methods=['POST'])
def answer():
    data = request.get_json()
    question = data.get('question', '')
    answers = engine.query(question)
    return jsonify({'answers': answers})

if __name__ == '__main__':
    
    app.run(debug=True)