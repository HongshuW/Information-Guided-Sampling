import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.gad_logits_processor_oracle import GrammarAlignedOracleLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, update_oracle_trie
import numpy as np
from tqdm import tqdm
import time

NUM_ITER = 30
EXPECTED_SIZE = 17
MODEL_ID = "TinyLlama/TinyLlama_v1.1" # pretrained llm
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
# TRIE_PATH = "tries/gad/binary_len_5_0_trie.json"
DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0
BATCH_SIZE = 1

@torch.inference_mode()
def inference_gad(model, tokenizer, prompt, grammar_str, trie):
    """
    latest version of gad test function prepared for run inference for iterations
    """
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, trie)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        gad_oracle_processor,
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

    generated_sequences = output.sequences  # Access the tensor of generated sequences
    # print(generated_sequences.shape)        # Print the shape of the sequences tensor
    # print(generated_sequences)

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    # Track whether each token is grammatically valid based on the initial grammar constraints
    acceptance_details_history = gad_oracle_processor.acceptance_details_history
    # Refined information after evaluating the grammatical validity of the sequence with additional adjustments (e.g., after sampling)
    adjusted_acceptance_details_history = gad_oracle_processor.adjusted_acceptance_details_history
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
    # print(f"grammar constrained generations: {generations}")
    return generated_tokens, acceptance_details_history,adjusted_acceptance_details_history, generations, metas, sum_log_prob

@torch.inference_mode()
def run_inference_gad_loading_trie(model, tokenizer):
    # Load EBNF grammar
    with open(GRAMMAR_PATH, "r") as file:
        grammar_str = file.read()
        print("grammar: ", grammar_str)

    # Tokenize prompt into ids
    prompt = "Generate a binary string of length 5"
    # Convert prompt into input IDs compatible with the model, return a pytorch tensor
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    start_time = time.time()
    adjusted_trie_before = Trie()
    adjusted_trie_after = Trie()
    outputs = []
    output_set = set()
    convergence = False
    for i in tqdm(range(NUM_ITER), desc="Running Inference"):
        """Draw sample from the LLM, incorporating trie and EFG"""
        generated_tokens, acceptance_details_history, adjusted_acceptance_details_history, generations, metas, sum_log_prob = inference_gad(model, tokenizer, prompt, grammar_str, adjusted_trie_before)
        # print(f"generated_tokens: {generated_tokens}, acceptance_details_history: {acceptance_details_history}")
        
        """
        Add sampled string to S:
        1. Update adjusted_trie_before using initial grammar acceptance details
        2. Computes updated_rate indicating how much the trie was modified.
        """
        _, updated_rate = update_oracle_trie(adjusted_trie_before, generated_tokens, acceptance_details_history)

        """
        Adjust EFG:
        Update adjusted_trie_before using refined acceptance details
        """
        update_oracle_trie(adjusted_trie_before, generated_tokens, adjusted_acceptance_details_history)

        """
        Update adjusted_trie_after with refined acceptance details
        """
        #update_oracle_trie(adjusted_trie_after, generated_tokens, adjusted_acceptance_details_history)

        result = {"answer": generations,
                  "sum_log_prob": sum_log_prob,
                  "metas": metas,
                  "updated_rate": updated_rate,
                  "prompt": prompt
                  }
        # print(f"result: {result}")
        # print(generations)
        outputs.append(generations)
        output_set.add(generations[0])
        if len(output_set) == EXPECTED_SIZE and not convergence:
            convergence = True
            print("convergence in ", i, "iterations")
            break

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")
    adjusted_trie_before.print_all_nodes()
    print(outputs)


if __name__ == "__main__":
    device = torch.device(DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.to(dtype=DTYPE)
    model.resize_token_embeddings(len(tokenizer))

    run_inference_gad_loading_trie(model, tokenizer)
    print("Inference Done!")
