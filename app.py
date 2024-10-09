from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from dotenv import load_dotenv

os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_name = "yx921/complaint_categorize_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

class_mapping = {
    0: "Academic grievance",
    2: "Behaviour grievance",
    1: "Facility grievance",
    3: "Finance grievance",
    4: "Other"
}

def predict_complaint(text):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)

    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = class_mapping.get(predicted_class, "Unknown")
    return predicted_label, probs.tolist(), predicted_class

@app.route('/categorize', methods=['POST'])
def predict():
    data = request.json
    if 'complaint' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400
    
    complaint_text = data['complaint']
    predicted_label, probabilities, predicted_class = predict_complaint(complaint_text)
    
    return jsonify({
        'complaint': complaint_text,
        'predicted_label': predicted_label,
        'predicted_class': predicted_class,
        'probabilities': probabilities,

    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)