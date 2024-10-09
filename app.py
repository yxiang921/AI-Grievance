from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model_name = "yx921/bert_complaint_classifier"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Define class mapping
class_mapping = {
    0: "Academic Complaint",
    1: "Facility Complaint",
    2: "Behaviour Complaint",
    3: "Finance Complaint",
    4: "Other"
}

# Prediction function
def predict_complaint(text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Move tensors to the appropriate device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Perform prediction
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)  # Get probabilities

    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1).item()
    predicted_label = class_mapping.get(predicted_class, "Unknown")
    return predicted_label, probs.tolist()

# Define API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'complaint' not in data:
        return jsonify({'error': 'No complaint text provided'}), 400
    
    complaint_text = data['complaint']
    predicted_label, probabilities = predict_complaint(complaint_text)
    
    return jsonify({
        'complaint': complaint_text,
        'predicted_label': predicted_label,
        'probabilities': probabilities
    })


@app.route('/hello', methods=['GET'])
def helloworld():
    return jsonify({'message': 'Hello World!'})


# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)