from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertModel
import torch
import os
from dotenv import load_dotenv
from flask_cors import CORS

os.environ["FLASK_ENV"] = "development"

app = Flask(__name__)
CORS(app)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

model_name = "yx921/complaint_categorize_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGINGFACE_TOKEN)

# bert_case_uncased = "bert-base-uncased"
# bert_tokenizer = BertTokenizer.from_pretrained(bert_case_uncased)
# bert_model = BertModel.from_pretrained(bert_case_uncased)

class_mapping = {
    0: "Academic",
    1: "Behaviour",
    2: "Facility",
    3: "Finance",
    4: "Other"
}

def categorize_grievance(grievance_text):
    inputs = tokenizer.encode_plus(
        grievance_text,
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
    predicted_label = class_mapping.get(predicted_class, "Unknown Category")
    return predicted_label, probs.tolist(), predicted_class


# def get_embedding(text):
#     inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state.mean(dim=1)


@app.route('/')
def index():
    print("Hello, World!")
    return "Hello, World!"


@app.route('/categorize', methods=['POST'])
def categorize():
    data = request.json
    if 'grievance' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400
    
    grievance_text = data['grievance']
    predicted_label, probabilities, predicted_class = categorize_grievance(grievance_text)
    
    return jsonify({
        'grievance': grievance_text,
        'predicted_label': predicted_label,
        'predicted_class': predicted_class,
        'probabilities': probabilities,
    })


# @app.route('/similiar', methods=['POST'])
# def similiarSearch():
    data = request.json
    if 'grievance' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400
    
    grievance_text = data['grievance']
    grievance_embedding = get_embedding(grievance_text)
    
    return jsonify({
        'grievance': grievance_text,
        'embedding': grievance_embedding.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)