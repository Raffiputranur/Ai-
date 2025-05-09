from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("indolem/indobert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("indolem/indobert-base-uncased")

texts = [
    "Pelayanan di restoran ini sangat memuaskan, makanannya juga enak",
    "Kecewa dengan kualitas produk ini, tidak sesuai ekspektasi saya"
]

encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**encoded_inputs)
    predictions = F.softmax(outputs.logits, dim=1)

for i, text in enumerate(texts):
    sentiment = "positif" if torch.argmax(predictions[i]) == 1 else "negatif"
    print(f"Text: {text}")
    print(f"Sentimen: {sentiment} (Confidence: {predictions[i].max().item():.4f})\n")