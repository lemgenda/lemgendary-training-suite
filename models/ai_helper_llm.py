"""Backward-compatible facade for models.ai_helper_llm.

Re-exports GenericCausalLLM and AIHelperLemGend from models.generic_causal_llm.
"""

from __future__ import annotations

from models.generic_causal_llm import AIHelperLemGend, GenericCausalLLM

__all__ = ["AIHelperLemGend", "GenericCausalLLM"]
