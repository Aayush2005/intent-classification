from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch

# Load model & tokenizer
model_path = "insurance_intent_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label map
label_map = {}
with open("label_map.txt") as f:
    for line in f:
        label, idx = line.strip().split(":")
        label_map[int(idx)] = label

def predict_intent(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    return label_map[predicted_class_id]
