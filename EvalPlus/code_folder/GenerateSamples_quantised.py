import subprocess
import codecarbon
import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import warnings
import csv

# install and import evalplus
# subprocess.check_call([sys.executable, "-m", "pip", "install", "evalplus", "--upgrade"])

# measure energy consumption
from codecarbon import EmissionsTracker

import evalplus

# as derived from the evalplus documentation
# https://github.com/evalplus/evalplus?tab=readme-ov-file#-quick-start
from evalplus.data import get_human_eval_plus, write_jsonl

# use StarCoder2 as LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# HumanEval: https://github.com/openai/human-eval?tab=readme-ov-file#usage
# EvalPlus: https://github.com/evalplus/evalplus/tree/master?tab=readme-ov-file#code-generation

def main(model_name, max_tokens, quantisation):
    # first, clear cuda memory
    torch.cuda.empty_cache()

    # ignore FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    tracker = EmissionsTracker(project_name="3B quantised", measure_power_secs=1, log_level='warning', output_dir='emissions_log')
    # Load the model

    # use 8 bit configuration
    if quantisation == '4bit':
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        # bnb_4bit_quant_type (str, optional, defaults to "fp4")
        # This sets the quantization data type in the bnb.nn.Linear4Bit layers.
        # nf4 outperforms fp4, as in QLoRA paper
    if quantisation == '8bit':
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # tracker.start_task("Load model")
    checkpoint = f"bigcode/starcoder2-{model_name}"
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, quantization_config=quantization_config, device_map={"":0}) # to GPU

    # print model device
    print("Model is stored on: ", next(model.parameters()).device)

    # tracker.stop_task()

    def GEN_SOLUTION(prompt, max_tokens):
        """Implement the GEN_SOLUTION function by calling the LLM to produce the complete solution (include the code) 
        and save the samples to samples.jsonl:"""
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        attention_mask = inputs.new_ones(inputs.shape)  # Create attention mask
        with torch.no_grad():
            outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=max_tokens, pad_token_id=tokenizer.eos_token_id)
        return tokenizer.decode(outputs[0])

    # Generate samples
    print(f"Model {checkpoint} loaded. Proceeding to generating samples.")
    tracker.start_task("Inference")

    start_time = time.time()
    samples = [
        dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"], max_tokens))
        for task_id, problem in get_human_eval_plus().items()
    ]

    # Save the samples to samples.jsonl
    write_jsonl(f"samples_quantised/samples_{model_name}_{max_tokens}_{quantisation}.jsonl", samples)

    tracker.stop_task()

    data = []

    end_time = round(time.time() - start_time, 2)
    print(f"Time used for starcoder2-{model_name}, quantised as {quantisation} with max. {max_tokens} new tokens is {end_time} seconds ({round(end_time/60, 1)} minutes).\nAverage of {round(end_time/162, 2)} seconds per coding task.")
    emissions = tracker.stop()
    print(f"The approximated emissions for the task with starcoder2-{model_name}, quantised as {quantisation} with max. {max_tokens} new tokens = {emissions} kgs of C02.")

    for task_name, task in tracker._tasks.items():
        # print(
        #     f"\nEmissions : {1000 * task.emissions_data.emissions} g CO₂ for task {task_name}"
        # )
        # print(
        #     f"\tEnergy : CPU {round(1000 * task.emissions_data.cpu_energy,3)} Wh. GPU {round(1000 * task.emissions_data.gpu_energy, 3)} Wh. RAM{round(1000 * task.emissions_data.ram_energy, 3)}Wh.\nTotal: {round(1000 * task.emissions_data.energy_consumed, 3)} Wh."
        # )
        # print(
        #     f"\tPower CPU:{task.emissions_data.cpu_power:.0f}W GPU:{task.emissions_data.gpu_power:.0f}W RAM{task.emissions_data.ram_power:.0f}W"
        #     + f" during {task.emissions_data.duration} seconds."
        # )
        data.append({
        'DRAM': round(1000 * task.emissions_data.ram_energy, 3),
        'GPU': round(1000 * task.emissions_data.gpu_energy, 3),
        'CPU': round(1000 * task.emissions_data.cpu_energy, 3),
        'total': round(1000 * task.emissions_data.energy_consumed, 3),
        'EstimatedCO2': round(1000 * task.emissions_data.emissions, 3),
        'Runtime': round(end_time, 1)
        })

    ### Write data to the CSV file ###
    csv_file = f'energy_data/results_{model_name}_{max_tokens}_{quantisation}.csv'

    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['DRAM', 'GPU', 'CPU', 'total', 'EstimatedCO2','Runtime'])
        if not file_exists:
            writer.writeheader()  # File doesn't exist yet, write the header
        writer.writerows(data)
    
    print("Data stored to csv.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give arguments for the dataset (humaneval or mbpp) and samples.')

    # Add the arguments
    parser.add_argument('--model_name', default='3b', type=str,
                        help='Chosen model: 3b/7b/15b. Provide number and small letter (e.g. 3b).')
    # parser.add_argument('--samples', default='samples/samples.json', type=str,
    #                     help='Filepath to samples.jsonl file.')
    parser.add_argument('--max_tokens', default=128, type=int,
                        help='Maximum number of tokens for the model.')
    parser.add_argument('--quantisation', default='8bit', type=str,
                        help='Quantise model weights in 4 or 8-bit. Provide as "4bit" or "8bit".')

    # Parse the arguments
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
