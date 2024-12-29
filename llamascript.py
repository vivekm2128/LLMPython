from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Replace with the path to your model directory
model_path = "/home/ubuntu/.llama/checkpoints/Llama3.2-1B"

# Load the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# Run the model on a sample input
input_text = "Hello, how can I assist you today?"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate text using the model
outputs = model.generate(**inputs, max_length=50)

# Decode and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
