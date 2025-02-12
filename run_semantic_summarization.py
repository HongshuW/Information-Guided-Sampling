import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.gad_logits_processor_oracle import GrammarAlignedOracleLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, update_oracle_trie
import numpy as np
from tqdm import tqdm
import time

# MODEL_ID = "TinyLlama/TinyLlama_v1.1" # pretrained llm
MODEL_ID = "meta-llama/Llama-3.2-1B"
DEVICE = "cuda"
# DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0
BATCH_SIZE = 1
PROMPT = "Summarize this sentence to context-free grammar:'generate a binary string of length 5 that ends with a 1'"

@torch.inference_mode()
def call_LLM(model, tokenizer, prompt):

    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
    ])

    # Generate
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]

    input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids,
        do_sample=True, # Enable probabilistic sampling
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=MAX_NEW_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
        temperature=TEMPERATURE,
        logits_processor=logits_processors,
        repetition_penalty=REPETITION_PENALTY,
        num_return_sequences=BATCH_SIZE, # Generate a batch of strings
        return_dict_in_generate=True,
        output_scores=True,
    )

    print()
    generated_sequences = output.sequences  # Access the tensor of generated sequences
    print(generated_sequences.shape)        # Print the shape of the sequences tensor
    print(generated_sequences)

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]

    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    transition_scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )

    metas = []
    sum_log_prob = 0
    for tok, score in zip(generated_tokens[0], transition_scores[0]):
        meta = {
            "token_id": int(tok),
            "token_str": tokenizer.decode(tok),
            "norm_score": float(score.cpu().numpy()),
            "prob": float(np.exp(score.cpu().numpy()))
        }
        metas.append(meta)
        sum_log_prob += float(score.cpu().numpy())

    return generated_tokens, generations, metas, sum_log_prob

@torch.inference_mode()
def run_semantic_summarization(model, tokenizer):
    # Tokenize prompt into ids
    prompt = PROMPT
    # Convert prompt into input IDs compatible with the model, return a pytorch tensor
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    start_time = time.time()
    """Draw sample from the LLM, incorporating trie and EFG"""
    generated_tokens, generations, metas, sum_log_prob = call_LLM(model, tokenizer, prompt)
    # print(f"generated_tokens: {generated_tokens}, acceptance_details_history: {acceptance_details_history}")
        
    print(generations)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")



if __name__ == "__main__":
    device = torch.device(DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model

    # Load and modify configuration
    config = AutoConfig.from_pretrained(MODEL_ID)
    # Adjust rope_scaling: keep only the required keys.
    # You can either set it manually:
    config.rope_scaling = {"type": "llama3", "factor": 32.0}

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, config=config)
    model.to(device)
    model.to(dtype=DTYPE)
    model.resize_token_embeddings(len(tokenizer))

    run_semantic_summarization(model, tokenizer)
