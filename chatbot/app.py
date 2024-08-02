from flask import Flask, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = Flask(__name__)

# huggingface_hub.login(os.getenv('TOKEN'))

model_name = os.getenv('MY_MODEL')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return prediction

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    if not user_question:
        return jsonify({'error': 'No se proporcionó ninguna pregunta.'}), 400

    # Realiza una predicción
    prediction = predict(user_question)
    return jsonify({'answer': f'The prediction is: {prediction}'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

