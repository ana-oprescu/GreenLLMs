# MasterThesis
This repository contains the code for my master thesis. 

##  Structure of repository

## Reproducing the results
Our results can be reproduced using the following steps. Note that we refer to job files which can be used on a cluster with slurm scheduling. Our code can be reproduced without these job files as well, in that case one should follow the sequence of steps from the job file.
1. First, install the conda environment with [install_env.job](https://github.com/PepijndeReus/MasterThesis/blob/main/job_files/install_env.job)
2. Check if the environment is installed with `conda activate thesis`

With this conda environment you can now rerun and/or edit the code.
For **regular StarCoder2-3B and StarCoder2-7B** we used [gen_samplesv2.job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samplesv2.job) which calls [GenerateSamplesv2.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/GenerateSamplesv2.py). The job file has documentation that can help navigate with various settings (e.g. creating data for just one model).

For the **quantised StarCoder2 models** you can either use the job files for specific settings, e.g. [StarCoder2-3B on 128 tokens](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised_3b_128.job) or the [generic quantisation job script](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/gen_samples_quantised.job). Likewise, code has been documented and some models are commented out. This can be changed according to the goal of the experiments such as performing the run over various nodes.

For the **Pythia experiments** we used the job files in the [Pythia folder](https://github.com/PepijndeReus/MasterThesis/tree/main/Pythia). 

Finally, for the **pruning experiments** we used [StarCoder_pruning.py](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/StarCoder_pruning.py), which is called by the [decoderlens job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/decoderlens.job) and evaluated in a separate [evaluation job](https://github.com/PepijndeReus/MasterThesis/blob/main/EvalPlus/evaluate_pruning.job).