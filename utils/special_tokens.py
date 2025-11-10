"""Utilities for registering LiveCC-specific special tokens."""

from __future__ import annotations

import logging
from typing import Iterable


logger = logging.getLogger(__name__)


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
        if not _resize_model_embeddings(model, len(tokenizer)):
            raise RuntimeError(
                "New special tokens were added but their embeddings could not be resized. "
                "Pass a model that exposes `resize_token_embeddings` or make sure the underlying "
                "language model (e.g. `model.thinker`) is provided."
            )
    return added


def _resize_model_embeddings(model, new_vocab_size: int) -> bool:
    """Attempt to resize token embeddings on the model or logical submodules."""

    resize_candidates = [
        model,
        getattr(model, "thinker", None),
        getattr(getattr(model, "thinker", None), "model", None),
    ]

    for candidate in resize_candidates:
        if candidate is None:
            continue
        resize_fn = getattr(candidate, "resize_token_embeddings", None)
        if resize_fn is None:
            continue
        try:
            resize_fn(new_vocab_size)
            return True
        except NotImplementedError:
            logger.debug(
                "resize_token_embeddings not supported for %s; trying next candidate",
                type(candidate).__name__,
            )
            continue

    logger.warning("Unable to resize embeddings for any candidate modules: %s", [type(c).__name__ for c in resize_candidates if c])
    return False
