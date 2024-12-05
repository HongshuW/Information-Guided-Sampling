# Grammar-Aligned Decoding with Faster Convergence

This project is a proof-of-concept tool for the *Information-Guided Sampling (IGS) algorithm*.

This repository is built from [transformers-GAD](https://github.com/ebmoon/transformers-GAD).



## Run IGS
### Step 0: clone this repo and create environment
Clone the repository.

Then create a new Conda environment using the provided requirements file.
```
conda create --name <env> python=3.11
conda activate <env>
pip install -r requirements.txt
pip install .
```

### Step 1: run the script
Execute the script:
```
python run_inference_gad.py
```

## Run baseline
The default algorithm is the Information-Guided Sampling (IGS) algorithm.

Change the `transformers_gad/generation/gad_logits_processor_oracle.py` file to run the *Adaptive Sampling with Approximate Expected Futures (ASAp) algorithm*:

Line 58:
Change from 
```
self.apply_oracle_adjustments_with_efi(acceptance, scores, current_parent)
```
to
```
self.apply_oracle_adjustments_gad(acceptance, scores, current_parent)
```

## Run with different configurations
Change configurations in `run_inference_gad.py` file.

Default configurations:
```
EXPECTED_SIZE = 17
NUM_ITER = EXPECTED_SIZE * 5
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0
BATCH_SIZE = 1
PROMPT = "Generate a binary string of length 5"
```