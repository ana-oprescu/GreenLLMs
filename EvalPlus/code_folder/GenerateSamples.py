import subprocess
import codecarbon
import sys
import os

# install and import evalplus
# subprocess.check_call([sys.executable, "-m", "pip", "install", "evalplus", "--upgrade"])

# measure energy consumption
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

import evalplus

# as derived from the evalplus documentation
# https://github.com/evalplus/evalplus?tab=readme-ov-file#-quick-start
from evalplus.data import get_human_eval_plus, write_jsonl

# use StarCoder2 as LLM
from transformers import AutoModelForCausalLM, AutoTokenizer
checkpoint = "bigcode/starcoder2-3b"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

def GEN_SOLUTION(prompt):
    """Implement the GEN_SOLUTION function by calling the LLM to produce the complete solution (include the code) 
        and save the samples to samples.jsonl:"""
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=101, truncation=True).to(device)
    attention_mask = inputs.new_ones(inputs.shape)  # Create attention mask
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0])

# HumanEval: https://github.com/openai/human-eval?tab=readme-ov-file#usage
# EvalPlus: https://github.com/evalplus/evalplus/tree/master?tab=readme-ov-file#code-generation

samples = [
    dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"]))
    for task_id, problem in get_human_eval_plus().items()
]
# TODO: Where to store the samples file? Evalplus/code_folder might be good idea
# Save the samples to samples.jsonl
# write_jsonl("samples_max_tokens_128.jsonl", samples)
write_jsonl("samples_max_tokens_512.jsonl", samples)
emissions = tracker.stop()
print(emissions)

# Should be executed as shell script (sanitize)
# print("import code")
# from evalplus import lecacy_sanitize
# print("run code")
# lecacy_sanitize.script("samples_max_tokens_128.jsonl")

# print("End of task")
