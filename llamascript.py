from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# Set to the path where your model is located
model_path = "/home/ubuntu/.llama/checkpoints/Llama3.2-1B"

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# Use the model
input_text = "Hello, how can I assist you today?"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)

# Print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
