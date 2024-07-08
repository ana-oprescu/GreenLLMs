import subprocess
import codecarbon
import sys
import os
import torch
import evalplus
import warnings
import time
import argparse
import csv

# install and import evalplus
# subprocess.check_call([sys.executable, "-m", "pip", "install", "evalplus", "--upgrade"])

from transformers import AutoModelForCausalLM, AutoTokenizer

# measure energy consumption
from codecarbon import EmissionsTracker
tracker = EmissionsTracker()
tracker.start()

# as derived from the evalplus documentation
# https://github.com/evalplus/evalplus?tab=readme-ov-file#-quick-start
from evalplus.data import get_human_eval_plus, write_jsonl
    
def main(model_name, max_tokens, idx):
    # first, clear cuda memory
    torch.cuda.empty_cache()

    # ignore FutureWarnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    tracker = EmissionsTracker(project_name="EvalPlus_Phi", measure_power_secs=1, log_level='warning', output_dir='emissions_log')
    # Load the model
    # tracker.start_task("Load model")
    torch.set_default_device("cuda")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

    # reduce idx layers of model
    model.model.layers = model.model.layers[:-idx]

    # tracker.stop_task()

    def GEN_SOLUTION(prompt, max_tokens):
        """Implement the GEN_SOLUTION function by calling the LLM to produce the complete solution (include the code) 
        and save the samples to samples.jsonl:"""
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)

        # added after jobID 241, prevent attention mask error
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        with torch.no_grad():
            outputs = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=max_tokens)
        return tokenizer.batch_decode(outputs)[0]

    # Generate samples
    print(f"\nPhi-2 loaded. Proceeding to generating samples with pruning the last {idx} layers.")
    tracker.start_task("Inference")

    start_time = time.time()
    samples = [
        dict(task_id=task_id, solution=GEN_SOLUTION(problem["prompt"], max_tokens))
        for task_id, problem in get_human_eval_plus().items()
    ]

    # Save the samples to samples.jsonl
    write_jsonl(f"samples_decode/samples_{model_name}_{max_tokens}_{idx}.jsonl", samples)

    tracker.stop_task()
    data = []

    end_time = round(time.time() - start_time, 2)
    print(f"Time used for {model_name} with max. {max_tokens} new tokens is {end_time} seconds ({round(end_time/60, 1)} minutes).\nAverage of {round(end_time/162, 2)} seconds per coding task.")
    emissions = tracker.stop()
    print(f"The approximated emissions for the task with {model_name} with max. {max_tokens} new tokens = {emissions} kgs of C02.")

    for task_name, task in tracker._tasks.items():
        data.append({
        'DRAM': round(1000 * task.emissions_data.ram_energy, 3),
        'GPU': round(1000 * task.emissions_data.gpu_energy, 3),
        'CPU': round(1000 * task.emissions_data.cpu_energy, 3),
        'total': round(1000 * task.emissions_data.energy_consumed, 3),
        'EstimatedCO2': round(1000 * task.emissions_data.emissions, 3),
        'Runtime': round(end_time, 1)
        })

    ### Write data to the CSV file ###
    csv_file = f'energy_data/pruning/{model_name}_{max_tokens}_minus_{idx}.csv'

    # Check if the file exists
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['DRAM', 'GPU', 'CPU', 'total', 'EstimatedCO2', 'Runtime'])
        if not file_exists:
            writer.writeheader()  # File doesn't exist yet, write the header
        writer.writerows(data)
    
    print(f"Data stored to csv: {csv_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Give arguments for the dataset (humaneval or mbpp) and samples.')

    # Add the arguments
    parser.add_argument('--model_name', default='Phi2', type=str,
                        help='Chosen model: Phi2.')
    # parser.add_argument('--samples', default='samples/samples.json', type=str,
    #                     help='Filepath to samples.jsonl file.')
    parser.add_argument('--max_tokens', default=128, type=int,
                        help='Maximum number of tokens for the model.')
    parser.add_argument('--idx', default=1, type=int,
                        help='how many layers should be removed.')

    # Parse the arguments
    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
