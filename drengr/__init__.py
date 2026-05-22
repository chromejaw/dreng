"""
drengr - Streamlined prompt dataset generation library

A production-ready tool for generating high-quality prompt datasets with 
industry-standard SOTA defaults. Single import, single function API.
"""

__version__ = "1.0.0"
__author__ = "drengr Team"

import os
import sys
import random
from typing import Optional, Union
from pathlib import Path

from .core.generator import generate_dataset_with_new_api
from .core.exceptions import DrengrError, ConfigurationError, GenerationError, ValidationError


def generate(
    total: int,
    *,
    output_path: Optional[str] = None,
    profile: str = "sota",
    seed: Optional[int] = None,
    embedding_backend: str = "auto",
    embedding_model: Optional[str] = None,
    use_llm_for_paraphrase: bool = True,
    llm_backend: str = "auto",
    include_golden: bool = True,
    preview: int = 0,
    stream: bool = False,
    overwrite: bool = False,
    run_ablation: bool = False,
    force: bool = False,
) -> str:
    """Generate and write dataset JSON. Returns path to file.
    
    Args:
        total: Number of prompts to generate
        output_path: Output file path (default: ./dreng_dataset_<total>_<seed>.json)
        profile: Generation profile ("sota" | "fast" | "cheap" | "dev")
        seed: Random seed for reproducibility
        embedding_backend: Embedding service ("auto" | "local" | "openai")
        embedding_model: Specific embedding model to use
        use_llm_for_paraphrase: Use LLM for paraphrase generation
        llm_backend: LLM service ("auto" | "local" | "openai" | "none")
        include_golden: Generate golden responses
        preview: Number of sample prompts to print (0=no preview)
        stream: Stream generation events/progress
        overwrite: Overwrite existing output file
        run_ablation: Run ablation experiments
        force: Bypass validation failures
        
    Returns:
        Path to generated dataset file
        
    Raises:
        DrengrError: Base exception for all drengr errors
        ConfigurationError: Configuration or setup errors
        GenerationError: Errors during dataset generation
        ValidationError: Validation failures
    """
    try:
        return generate_dataset_with_new_api(
            total=total,
            output_path=output_path,
            profile=profile,
            seed=seed,
            embedding_backend=embedding_backend,
            embedding_model=embedding_model,
            use_llm_for_paraphrase=use_llm_for_paraphrase,
            llm_backend=llm_backend,
            include_golden=include_golden,
            preview=preview,
            stream=stream,
            overwrite=overwrite,
            run_ablation=run_ablation,
            force=force,
        )
    except Exception as e:
        if isinstance(e, DrengrError):
            raise
        else:
            raise GenerationError(f"Unexpected error during generation: {e}") from e


# Export only the generate function
__all__ = ["generate"]