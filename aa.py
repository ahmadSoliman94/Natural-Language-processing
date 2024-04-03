import torch
import transformers

from transformers import LlamaForCausalLM, LlamaTokenizer

model_dir = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_dir)

tokenizer = LlamaTokenizer.from_pretrained(model_dir)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
sequences = pipeline(
    "I have tomatoes, basil and cheese at home. What can I cook for dinner?\n",
    do_sample=True,
    top_k=10,
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=200,
)
print("HI")
for seq in sequences:
    print(f"{seq['generated_text']}")
