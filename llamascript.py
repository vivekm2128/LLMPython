from llama_cpp import Llama

# define n_ctx manually to permit larger contexts
LLM = Llama(model_path="/home/ubuntu/.llama/checkpoints/Llama3.2-1B/tokenizer.model", n_ctx=512)

# create a text prompt
prompt = "Tell me why life is so hard in 5 sentances?"

# set max_tokens to 0 to remove the response size limit
output = LLM(prompt, max_tokens=0)

# display the response
print(output["choices"][0]["text"])
