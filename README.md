# MasterThesis
This repository contains the code for my master thesis: Exploring energy consumption, inference time and accuracy in code Large Language Models. The thesis is available in [this GitHub repository](https://github.com/PepijndeReus/MasterThesis/blob/main/MScThesis_PepijndeReus.pdf). All files should have sufficient documentation for reproduction and understanding. Any remaining questions or comments may be raised via an issue in this repository or sent to [my e-mail](mailto:p.dereus@uva.nl).

**Abstract** 
Next to a significant increase in users, AI also has an increasing energy consumption. Microsoft and Google reported 30.9%, respectively 48% more carbon emissions last year due to increased energy requirements for AI. To reduce the dangerous consequences of climate change, the Paris Agreement aims to limit global warming to 1.5 degrees by limiting carbon emissions. Apart from ChatGPT, Large Language Models (LLMs) are widely adopted in the programming community with plug-ins such as GitHub Copilot. As very little work describes the energy consumption of code LLMs during inference, we formulate the research question: how can we reduce the energy consumption of code LLMs to limit global warming? In this thesis, we aim to reduce the energy consumption of StarCoder2 models with minimal harm to accuracy by reducing training time, compressing weights via quantisation and pruning the last layers. Our experiments indicate that none of our experiments succeeds in reducing energy consumption without compromising accuracy. Nevertheless, we see possibilities and provide suggestions to optimise compressing weights by quantisation.

##  Structure of repository
The repository has three folders:
1. [EvalPlus](https://github.com/PepijndeReus/MasterThesis/tree/main/EvalPlus)\This folder contains all code and job scripts for the experiments on pruning & quantisation.
2. [Pythia](https://github.com/PepijndeReus/MasterThesis/tree/main/Pythia)\This folder contains all code and job scripts for the experiments with checkpoints ([Pythia 2.8B](https://huggingface.co/EleutherAI/pythia-2.8b))
3. [Requirements](https://github.com/PepijndeReus/MasterThesis/tree/main/job_files)\This folder contains the .yaml file for the conda environment and a script to install the environment.

## Reproducing the results
Our results can be reproduced using the following steps. Note that we refer to job files which can be used on a cluster with slurm scheduling. Our code can be reproduced without these job files as well, in that case one should follow the sequence of steps from the job file.
1. First, install the conda environment with [install_env.job](https://github.com/PepijndeReus/MasterThesis/blob/main/job_files/install_env.job)
2. Check if the environment is installed with `conda activate thesis`

With this conda environment you can now rerun and/or edit the code.
For **regular StarCoder2-3B and StarCoder2-7B** we used [gen_samplesv2.job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samplesv2.job) which calls [GenerateSamplesv2.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/GenerateSamplesv2.py). The job file has documentation that can help navigate with various settings (e.g. creating data for just one model).

For the **quantised StarCoder2 models** you can either use the job files for specific settings, e.g. [StarCoder2-3B on 128 tokens](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised_3b_128.job) or the [generic quantisation job script](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised.job). Likewise, code has been documented and some models are commented out. This can be changed according to the goal of the experiments such as performing the run over various nodes.

For the **Pythia experiments** we used the job files in the [Pythia folder](https://github.com/PepijndeReus/MasterThesis/tree/main/Pythia). 

Finally, for the **pruning experiments** we used [StarCoder_pruning.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/StarCoder_pruning.py), which is called by the [decoderlens job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/decoderlens.job) and evaluated in a separate [evaluation job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/evaluate_pruning.job).