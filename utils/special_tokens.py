"""Utilities for registering LiveCC-specific special tokens."""

from __future__ import annotations

from typing import Iterable


REACTION_SPECIAL_TOKENS: tuple[str, ...] = (
    "<emotion>",
    "</emotion>",
    "<reaction>",
    "</reaction>",
    "<think>",
    "</think>",
)


def ensure_special_tokens(tokenizer_or_processor, model=None, extra_tokens: Iterable[str] | None = None) -> int:
    """Ensure tokenizer (or processor.tokenizer) knows LiveCC reaction tokens.

    Args:
        tokenizer_or_processor: A HF tokenizer or processor instance.
        model: Optional HF model to resize when new tokens are added.
        extra_tokens: Optional iterable of additional tokens to register.

    Returns:
        Number of tokens newly added to the tokenizer vocabulary.
    """

    tokenizer = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
    if tokenizer is None:
        raise ValueError("Tokenizer or processor with .tokenizer is required")

    tokens = list(REACTION_SPECIAL_TOKENS)
    if extra_tokens:
        tokens.extend(extra_tokens)

    special_tokens = {"additional_special_tokens": tokens}
    added = tokenizer.add_special_tokens(special_tokens)
    if added > 0 and model is not None:
        model.resize_token_embeddings(len(tokenizer))
    return added
