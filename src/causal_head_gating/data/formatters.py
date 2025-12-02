"""
Few-shot prompt formatting utilities.

This module replicates the original `format_data.py` functionality for creating
few-shot prompts with proper train/validation/example splits.
"""

import itertools
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import torch
from torch import Tensor

from .tokenization import PromptTokenizer


def generate_prompt_permutations(
    prompt: str,
    options: Dict[str, int],
    max_permutations: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Generate all permutations of multiple choice options.

    For MCQ tasks, this creates different orderings of the answer choices
    to prevent the model from learning positional biases.

    Args:
        prompt: The question text.
        options: Dict mapping option text to correctness (1 for correct, 0 for wrong).
        max_permutations: Maximum number of permutations to generate.

    Returns:
        List of dicts with 'question', 'options', and 'target' keys.
    """
    items = list(options.items())
    all_perms = list(itertools.permutations(items))
    perms = (
        random.sample(all_perms, max_permutations)
        if max_permutations is not None and max_permutations < len(all_perms)
        else all_perms
    )
    results = []
    for perm in perms:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        labeled = [f"{letters[i]}. {k}" for i, (k, _) in enumerate(perm)]
        correct_index = next(i for i, (_, v) in enumerate(perm) if v == 1)
        results.append({
            "question": prompt.strip(),
            "options": "\n".join(labeled),
            "target": letters[correct_index],
        })
    return results


def assign_split(
    df_questions: pd.DataFrame,
    num_examples: int,
    f_validation: float = 0.1,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Assign train/validation/example splits to questions.

    The shortest questions (by total length) are assigned to the 'example' split
    to be used as few-shot examples. Remaining questions are split between
    'train' and 'validation'.

    Args:
        df_questions: DataFrame with 'question' and 'target' columns.
        num_examples: Number of questions to reserve as few-shot examples.
        f_validation: Fraction of remaining questions for validation.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added 'split' column ('example', 'train', or 'validation').
    """
    df = df_questions.copy()
    rng = random.Random(seed)

    # Calculate total length for each question
    if "options" in df.columns:
        grouped = df.groupby("question").agg(
            {"options": lambda x: sum(len(opt) for opt in x)}
        )
        grouped["total_len"] = grouped.index.str.len() + grouped["options"]
    elif "target" in df.columns:
        grouped = df.groupby("question").agg({"target": lambda x: len(x.iloc[0])})
        grouped["total_len"] = grouped.index.str.len() + grouped["target"]
    else:
        raise ValueError("DataFrame must contain either 'options' or 'target' column.")

    # Shortest questions become examples
    shortest = grouped.sort_values("total_len").head(num_examples).index
    df["split"] = "train"
    df.loc[df["question"].isin(shortest), "split"] = "example"

    # Split remaining into train/validation
    remaining = df.loc[~df["question"].isin(shortest), "question"].unique()
    num_val = int(len(remaining) * f_validation)
    val_qs = set(rng.sample(list(remaining), num_val))
    df.loc[df["question"].isin(val_qs), "split"] = "validation"

    return df


def create_prompts(
    df_questions: pd.DataFrame,
    num_examples: int,
    format_example_fn: Callable[[Dict[str, Any]], str],
    instructions: Optional[str] = None,
    seed: Optional[int] = None,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n",
) -> pd.DataFrame:
    """
    Create few-shot prompts from a DataFrame of questions.

    Each prompt will contain:
    1. Optional instructions
    2. N randomly sampled examples (from the 'example' split)
    3. The actual question to answer

    Args:
        df_questions: DataFrame with 'question', 'target', and 'split' columns.
        num_examples: Number of few-shot examples to include in each prompt.
        format_example_fn: Function to format each example as a string.
        instructions: Optional instruction text to prepend.
        seed: Random seed for reproducibility.
        question_prefix: Prefix for questions (e.g., "Question: ").
        answer_prefix: Prefix for answers (e.g., "Answer: ").
        sep: Separator between examples.

    Returns:
        DataFrame with 'split', 'prompt', and 'target' columns.
    """
    rng = random.Random(seed)
    df = df_questions.copy()

    # Group examples by question
    example_groups = df[df["split"] == "example"].groupby("question")
    example_dict = {q: g.to_dict("records") for q, g in example_groups}

    def build_prompt(row: pd.Series) -> Tuple[str, str]:
        # Sample examples
        sampled = rng.sample(list(example_dict), min(num_examples, len(example_dict)))
        examples = [rng.choice(example_dict[q]) for q in sampled]
        example_strs = [format_example_fn(ex) for ex in examples]

        # Build prompt
        text = (instructions + "\n") if instructions else ""
        text += sep.join(example_strs)

        # Add the actual question
        query_lines = [f"{question_prefix}{row['question']}"]
        if "options" in row and pd.notna(row.get("options")):
            query_lines.append(row["options"])
        query_lines.append(f"{answer_prefix}")
        query_block = "\n".join(query_lines)

        return sep.join([text, query_block]), row["target"]

    records = []
    for split in ["train", "validation"]:
        for _, row in df[df["split"] == split].iterrows():
            prompt, target = build_prompt(row)
            records.append({"split": split, "prompt": prompt, "target": target})

    return pd.DataFrame.from_records(records)


def create_fewshot_dataset(
    tokenizer: Any,
    questions: List[str],
    targets: Union[List[str], List[Dict[str, int]]],
    instructions: Optional[str] = None,
    max_permutations: Optional[int] = None,
    example_set_size: int = 50,
    num_examples: int = 1,
    f_validation: float = 0.1,
    seed: int = 0,
    question_prefix: str = "Question: ",
    answer_prefix: str = "Answer: ",
    sep: str = "\n",
    verbose: bool = True,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Tensor, Tensor]:
    """
    Create a few-shot learning dataset with train/validation splits.

    This replicates the original `create_dataset()` function from `format_data.py`.
    It:
    1. Assigns the shortest questions to be few-shot examples
    2. Creates prompts with randomly sampled examples for each train/val question
    3. Tokenizes everything with proper loss masks

    Supports both free-response questions (FRQ) and multiple choice questions (MCQ).

    Args:
        tokenizer: HuggingFace tokenizer.
        questions: List of question strings.
        targets: Either:
            - List of target/answer strings (FRQ mode)
            - List of dicts mapping options to correctness {option: 0 or 1} (MCQ mode)
        instructions: Optional instruction text to prepend to all prompts.
        max_permutations: For MCQ, maximum answer orderings per question.
        example_set_size: Number of questions to reserve as few-shot examples.
        num_examples: Number of examples to include in each prompt.
        f_validation: Fraction of non-example questions for validation.
        seed: Random seed for reproducibility.
        question_prefix: Prefix for questions.
        answer_prefix: Prefix for answers.
        sep: Separator between examples.
        verbose: Whether to print progress.
        device: Device to place tensors on.

    Returns:
        Tuple of (df_prompts, input_ids, loss_masks):
        - df_prompts: DataFrame with 'split', 'prompt', 'target' columns
        - input_ids: Tensor of shape [num_samples, seq_len]
        - loss_masks: Boolean tensor of shape [num_samples, seq_len]

    Example (FRQ - math):
        >>> df_prompts, input_ids, loss_masks = create_fewshot_dataset(
        ...     tokenizer,
        ...     questions=problems,
        ...     targets=solutions,
        ...     instructions="Solve step by step.",
        ...     example_set_size=50,
        ...     num_examples=1,
        ... )

    Example (MCQ):
        >>> targets = [{"Paris": 1, "London": 0, "Berlin": 0}]
        >>> df_prompts, input_ids, loss_masks = create_fewshot_dataset(
        ...     tokenizer,
        ...     questions=["What is the capital of France?"],
        ...     targets=targets,
        ...     max_permutations=6,
        ... )
    """
    prompt_tokenizer = PromptTokenizer(tokenizer, device=device)

    # Handle MCQ (dict targets) vs FRQ (string targets)
    is_mcq = len(targets) > 0 and isinstance(targets[0], dict)

    if is_mcq:
        # Generate permutations for multiple choice questions
        all_questions = []
        for q, opts in zip(questions, targets):
            all_questions.extend(generate_prompt_permutations(q, opts, max_permutations))
        if verbose:
            print(f"Generated {len(all_questions)} MCQ permutations.")
        df_questions = pd.DataFrame(all_questions)
    else:
        # Free-response questions
        df_questions = pd.DataFrame({"question": questions, "target": targets})

    df_questions = df_questions.drop_duplicates().reset_index(drop=True)

    # Assign splits
    df_questions = assign_split(
        df_questions,
        num_examples=example_set_size,
        f_validation=f_validation,
        seed=seed,
    )

    if verbose:
        print(
            f"Assigned splits: {df_questions['split'].value_counts().to_dict()}. "
            "Generating few-shot prompts."
        )

    # Create few-shot prompts with appropriate format function
    if is_mcq:
        # MCQ format includes options
        format_fn = lambda ex: (
            f"{question_prefix}{ex['question']}\n"
            f"{ex['options']}\n"
            f"{answer_prefix}{ex['target']}"
        )
    else:
        # FRQ format
        format_fn = lambda ex: (
            f"{question_prefix}{ex['question']}\n"
            f"{answer_prefix}{ex['target']}"
        )

    df_prompts = create_prompts(
        df_questions,
        num_examples=num_examples,
        format_example_fn=format_fn,
        instructions=instructions,
        seed=seed,
        question_prefix=question_prefix,
        answer_prefix=answer_prefix,
        sep=sep,
    )

    if verbose:
        print(f"Generated {len(df_prompts)} unique prompts. Tokenizing.")

    # Tokenize
    input_ids, loss_masks = prompt_tokenizer.tokenize_batch(
        df_prompts["prompt"].tolist(),
        df_prompts["target"].tolist(),
    )

    if verbose:
        # Find actual lengths (before padding)
        lengths = (input_ids != tokenizer.pad_token_id).sum(dim=1)
        print(
            f"Tokenized {len(input_ids)} prompts with "
            f"min length {lengths.min().item()} and max length {lengths.max().item()}."
        )

    return df_prompts, input_ids, loss_masks


def load_math_dataset(
    tokenizer: Any,
    dataset_name: str = "nvidia/OpenMathInstruct-2",
    split: str = "train_1M",
    num_examples: int = 50,
    num_train: int = 50000,
    num_validation: int = 5000,
    instructions: Optional[str] = None,
    num_fewshot_examples: int = 1,
    seed: int = 0,
    lengths_cache_path: Optional[str] = None,
    verbose: bool = True,
    device: str = "cpu",
) -> Tuple[pd.DataFrame, Tensor, Tensor]:
    """
    Load and prepare the OpenMathInstruct-2 dataset for CHG training.

    This function automates the full original math notebook workflow:
    1. Load dataset from HuggingFace
    2. Compute sequence lengths (or load from cache)
    3. Filter to problems with non-empty answers
    4. Sort by length and select shortest problems
    5. Create few-shot prompts with `create_fewshot_dataset()`

    Args:
        tokenizer: HuggingFace tokenizer.
        dataset_name: HuggingFace dataset name.
        split: Dataset split to use.
        num_examples: Number of shortest problems to use as few-shot examples.
        num_train: Number of training problems (after examples).
        num_validation: Number of validation problems.
        instructions: Instruction text for prompts. If None, uses default math instructions.
        num_fewshot_examples: Number of few-shot examples per prompt.
        seed: Random seed for reproducibility.
        lengths_cache_path: Path to save/load precomputed lengths (parquet file).
            If file exists, loads from cache. If None, computes but doesn't cache.
        verbose: Whether to print progress.
        device: Device to place tensors on.

    Returns:
        Tuple of (df_prompts, input_ids, loss_masks) ready for CHG training.

    Length Selection:
        The function selects the **shortest** problems from the dataset. The total
        number of problems selected is `num_examples + num_train + num_validation`.

        With the default parameters (50 + 50,000 + 5,000 = 55,050), this matches the
        original paper's approach of selecting the 55,050 shortest problems, which
        typically results in sequences of ~170-340 tokens.

    Example:
        >>> # Full workflow in one call
        >>> df_prompts, input_ids, loss_masks = load_math_dataset(
        ...     tokenizer,
        ...     num_train=50000,
        ...     lengths_cache_path="math_lengths.parquet",
        ... )

        >>> # Create dataset for training
        >>> train_mask = df_prompts['split'] == 'train'
        >>> train_dataset = MaskedSequenceDataset(
        ...     pad_token_id=tokenizer.pad_token_id,
        ...     input_ids=input_ids[train_mask.values],
        ...     loss_masks=loss_masks[train_mask.values],
        ... )

    Note:
        Computing lengths for 1M samples takes significant time (~30-60 minutes).
        Use `lengths_cache_path` to save and reuse computed lengths.
    """
    from pathlib import Path

    from datasets import load_dataset
    from tqdm.auto import tqdm

    # Default instructions match original notebook
    if instructions is None:
        instructions = (
            "For each problem, explain your reasoning step by step and use LaTeX "
            "for all mathematical expressions. Indicate your final answer using \\boxed{...}."
        )

    # Define the number of shortest problems (by token length) to select from the dataset.
    # Defaults to 50 + 50,000 + 5,000 = 55,050, matching the number used in the paper.
    total_needed = num_examples + num_train + num_validation

    # Check for cached lengths
    cache_path = Path(lengths_cache_path) if lengths_cache_path else None
    if cache_path and cache_path.exists():
        if verbose:
            print(f"Loading precomputed lengths from {cache_path}")
        df_problems = pd.read_parquet(cache_path)
    else:
        # Load dataset from HuggingFace
        if verbose:
            print(f"Loading dataset {dataset_name} ({split})...")
        hf_dataset = load_dataset(dataset_name, split=split)

        # Compute lengths for all samples
        if verbose:
            print(f"Computing sequence lengths for {len(hf_dataset)} samples...")
            print("(This may take a while. Use lengths_cache_path to save results.)")

        prompt_tokenizer = PromptTokenizer(tokenizer, device=device)

        rows = []
        iterator = tqdm(hf_dataset) if verbose else hf_dataset
        for item in iterator:
            problem = item["problem"]
            solution = item["generated_solution"]
            answer = item.get("expected_answer", "")

            # Compute tokenized length
            text_tokens, _ = prompt_tokenizer.tokenize(problem, solution)
            rows.append({
                "problem": problem,
                "solution": solution,
                "answer": answer,
                "length": len(text_tokens),
            })

        df_problems = pd.DataFrame(rows)

        # Save to cache if path provided
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df_problems.to_parquet(cache_path)
            if verbose:
                print(f"Saved lengths to {cache_path}")

    # Filter and select shortest problems (matching original notebook)
    if verbose:
        print("Filtering and selecting shortest problems...")

    # Filter to problems with non-empty answers
    df_problems = df_problems[df_problems["answer"].str.len() > 0]

    # Drop duplicate problems
    df_problems = df_problems.drop_duplicates("problem")

    # Sort by length and take shortest
    df_problems = df_problems.sort_values("length").head(total_needed).reset_index(drop=True)

    if verbose:
        print(f"Selected {len(df_problems)} shortest problems")
        print(f"  Length range: {df_problems['length'].min()} - {df_problems['length'].max()}")

    # Create few-shot dataset
    df_prompts, input_ids, loss_masks = create_fewshot_dataset(
        tokenizer=tokenizer,
        questions=df_problems["problem"].tolist(),
        targets=df_problems["solution"].tolist(),
        instructions=instructions,
        example_set_size=num_examples,
        num_examples=num_fewshot_examples,
        f_validation=num_validation / (num_train + num_validation),
        seed=seed,
        verbose=verbose,
        device=device,
    )

    return df_prompts, input_ids, loss_masks
