import torch
import json
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.gad_logits_processor_oracle import GrammarAlignedOracleLogitsProcessor
from transformers_gad.build_oracle.build_oracle_trie import Trie, update_oracle_trie
import os
import numpy as np
import json
from tqdm import tqdm
import time


NUM_ITER = 10
MODEL_ID = "TinyLlama/TinyLlama_v1.1" # pretrained llm
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = "tries/binary_len_5_0_trie.json"
DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

def load_oracle_trie(trie_file):
    with open(trie_file, 'rb') as f:
        trie = pickle.load(f)
    return trie

def construct_gad_output_file_path(args):
    model_name = args.model_id.split("/")[-1]
    grammar_prompt_file = args.grammar_prompt_file.split("/")[-1]
    grammar_prompt_name = grammar_prompt_file.split(".")[0]
    output_file_path = os.path.join(args.output_folder,
                                    f"gad_g-{grammar_prompt_name}_{model_name}_p-{args.prompt_type}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

def construct_gad_output_file_path_from_folder(args, test_filename):
    model_name = args.model_id.split("/")[-1]
    output_file_path = os.path.join(args.output_folder, f"{test_filename}")
    output_file_path = os.path.join(output_file_path, f"gad_{model_name}_i{args.iter}_{args.device}_sd{args.seed}_{args.dtype}.jsonl")
    output_directory = os.path.dirname(output_file_path)
    # Ensure the directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    return output_file_path

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
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
    )

    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    acceptance_details_history = gad_oracle_processor.acceptance_details_history
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
    for i in tqdm(range(NUM_ITER), desc="Running Inference"):
        generated_tokens, acceptance_details_history, adjusted_acceptance_details_history, generations, metas, sum_log_prob = inference_gad(model, tokenizer, prompt, grammar_str, adjusted_trie_before)
        # print(f"generated_tokens: {generated_tokens}, acceptance_details_history: {acceptance_details_history}")
        _, updated_rate = update_oracle_trie(adjusted_trie_before, generated_tokens, acceptance_details_history)
        update_oracle_trie(adjusted_trie_before, generated_tokens, adjusted_acceptance_details_history)
        update_oracle_trie(adjusted_trie_after, generated_tokens, adjusted_acceptance_details_history)

        result = {"answer": generations,
                  "sum_log_prob": sum_log_prob,
                  "metas": metas,
                  "updated_rate": updated_rate,
                  "prompt": prompt
                  }
        print(f"result: {result}")

        json_record = json.dumps(result)

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")


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
    print("GAD Inference Done!")
