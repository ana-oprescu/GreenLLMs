# An exploration of the effect of quantisation on energy consumption and inference time of Code Large Language Models
This repository contains the code for my master thesis: Exploring energy consumption, inference time and accuracy in code Large Language Models. The pre-print is available [here](). All files should have sufficient documentation for reproduction and understanding. Any remaining questions or comments may be raised via an issue in this repository or sent to [my e-mail](mailto:p.dereus@uva.nl).
This repository consists of three parts:
1. [Background information on the paper](##-Background-information-on-the-paper)
2. [Structure of this repository](##-Structure-of-repository)
3. [Reproduction of the results](##-Reproducing-the-results)

**Abstract** \
This study examines quantisation and pruning strategies to reduce energy consumption in code Large Language Models (LLMs) inference. Using StarCoder2, we observe increased energy demands with quantization due to lower throughput and some accuracy losses. Conversely, pruning reduces energy usage but impairs performance. The results highlight challenges and trade-offs in LLM model compression. We suggest future work on hardware-optimized quantization to enhance efficiency with minimal loss in accuracy.

## Background information on the paper

### StarCoder2 performance
In the StarCoder2 paper, the authors report an average pass@1 of 31.7 on HumanEval and 27.4 on HumanEval+ for StarCoder2-3B. For StarCoder2-7B, the pass@1 scores are 35.4 and 29.9 respectively. Table II shows our pass@1 scores of the StarCoder2-3B \& StarCoder2-7B models with different limits to the tokens generated. Our results for StarCoder2-3B are thus 18.3 and 17 points lower for HumanEval and HumanEval+ respectively. For StarCoder2-7B our results are 27.5 points lower on HumanEval and 23.8 lower on HumanEval+. The authors report that the model predicts 128 new tokens for each prompt, which we increased to 256 resulting in 2.5 points increase on HumanEval+ for StarCoder2-3B and a 2.4 points increase on HumanEval+ for StarCoder2-7B. Further increasing the maximum amount of tokens to 512 did not lead to improvements compared to 256 tokens and therefore is left out. We reached out to the StarCoder2 team, providing our code in the attachment for comparison, asking what might caused this deviation but unfortunately never received a reply. To determine our framework's correctness, we also analysed Phi-2 on the HumanEval+ benchmark. We found that the pass@1 score for Phi-2 matched the reported score in the paper, these are available [here]().

Finally, another reason why the pass@1 is rather low is that the StarCoder2 models are not instruction-tuned. Instruction-tuned models are optimised to follow instructions such as the HumanEval+ tasks. An example is ChatGPT, the GPT-3 model itself merely predicts the next token but ChatGPT has been instructed to ask questions and react to input to behave as a chatbot. In this case, the StarCoder2 authors report that they are at a disadvantage compared to other models which are instruction-tuned. Our results indicate that the model suffers from echoing the prompt and not providing code that answers the prompt. Though the instruction tuning partly explains the lower pass@1 score for related small models on HumanEval+, it does not explain why the results from the StarCoder2 paper are not reproducible. Table II shows us that StarCoder2-3B outperforms StarCoder2-7B on the HumanEval+ task by 4.2 points. This halving of the pass@1 is unexpected as in the paper StarCoder2-7B marginally outperforms StarCoder2-3B. Because the authors did not report hyperparameter settings, we cannot find the cause of this difference. With StarCoder2-7B about twice the size of StarCoder2-3B, we would expect a higher pass@1. However, where StarCoder2-3B is trained on the 17 most used programming languages, StarCoder2-7B is trained on 619 programming languages [Lozhkov (2024)](https://arxiv.org/pdf/2402.19173, it is plausible that the model predicted more irrelevant code.

### Cache misses
For 4-bit quantisation L1-misses increased by 75\%, L2-misses increased by 45\%. With 8-bit quantisation, L1-misses increased by 315\% and L2-misses by 435\%. We show these results in the Table below:


|            | **4-bit L1 misses** | **4-bit L2 misses** | **8-bit L1 misses** | **8-bit L2 misses** |
|------------|----------------------|----------------------|----------------------|----------------------|
| **Run 1**  | 77%                 | 45%                 | 322%                | 428%                |
| **Run 2**  | 77%                 | 50%                 | 315%                | 439%                |
| **Run 3**  | 73%                 | 51%                 | 302%                | 437%                |
| **Run 4**  | 75%                 | 47%                 | 316%                | 430%                |
| **Run 5**  | 75%                 | 48%                 | 318%                | 442%                |
| **Avg. increase** | **75%**       | **48%**             | **315%**            | **435%**            |

*The increased L1 and L2 cache misses for each of the five runs in experiment 2 for StarCoder2-7B on 128 tokens.*



### Pruning extended

### Phi-2 performance



##  Structure of repository
The repository has three folders:
1. [EvalPlus](https://github.com/PepijndeReus/MasterThesis/tree/main/EvalPlus) \
This folder contains all code and job scripts for the experiments on pruning & quantisation.
2. [Pythia](https://github.com/PepijndeReus/MasterThesis/tree/main/Pythia) \
This folder contains all code and job scripts for the experiments with checkpoints ([Pythia 2.8B](https://huggingface.co/EleutherAI/pythia-2.8b))
3. [Requirements](https://github.com/PepijndeReus/MasterThesis/tree/main/job_files) \
This folder contains the .yaml file for the conda environment and a script to install the environment.

## Reproducing the results
Our results can be reproduced using the following steps. Note that we refer to job files which can be used on a cluster with slurm scheduling. Our code can be reproduced without these job files as well, in that case one should follow the sequence of steps from the job file.
1. First, install the conda environment with [install_env.job](https://github.com/PepijndeReus/MasterThesis/blob/main/job_files/install_env.job)
2. Check if the environment is installed with `conda activate thesis`

With this conda environment you can now rerun and/or edit the code.
For **regular StarCoder2-3B and StarCoder2-7B** we used [gen_samplesv2.job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samplesv2.job) which calls [GenerateSamplesv2.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/GenerateSamplesv2.py). The job file has documentation that can help navigate with various settings (e.g. creating data for just one model).

For the **quantised StarCoder2 models** you can either use the job files for specific settings, e.g. [StarCoder2-3B on 128 tokens](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised_3b_128.job) or the [generic quantisation job script](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised.job). Likewise, code has been documented and some models are commented out. This can be changed according to the goal of the experiments such as performing the run over various nodes.

For the **Pythia experiments** we used the job files in the [Pythia folder](https://github.com/PepijndeReus/MasterThesis/tree/main/Pythia). 

Finally, for the **pruning experiments** we used [StarCoder_pruning.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/StarCoder_pruning.py), which is called by the [decoderlens job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/decoderlens.job) and evaluated in a separate [evaluation job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/evaluate_pruning.job).
