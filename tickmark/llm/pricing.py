"""Per-token list prices, used to put a dollar figure on every request.

Prices are USD per million tokens, standard (non-batch, non-cached) rates, as listed in
October 2026. Provider prices change: verify them before relying on the cost numbers,
and override any model with TICKMARK_PRICE_<MODEL>="input,output", for example
TICKMARK_PRICE_GPT_5_5="5.00,30.00". A model with no known price is reported as
unpriced rather than guessed.
"""

from __future__ import annotations

import os
import re

from tickmark.models import Usage

PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts up to 100K tokens; Tickmark's prompts stay below that
    "text-embedding-3-small": (0.02, 0.0),
    # TypeSafe Jev bills input tokens only.
    "jev-latest": (0.042, 0.0),
    "jev-1.14.0": (0.042, 0.0),
    "jev-1.13.0": (0.042, 0.0),
}


def _override(model: str) -> tuple[float, float] | None:
    key = "TICKMARK_PRICE_" + re.sub(r"[^A-Z0-9]", "_", model.upper())
    value = os.environ.get(key)
    if not value:
        return None
    input_price, _, output_price = value.partition(",")
    return float(input_price), float(output_price or 0)


def price_for(model: str) -> tuple[float, float] | None:
    return _override(model) or PRICES_PER_MTOK.get(model)


def usage_for(model: str, input_tokens: int, output_tokens: int) -> Usage:
    price = price_for(model)
    if price is None:
        return Usage(input_tokens, output_tokens, 0.0, calls=1, unpriced_calls=1)
    cost = (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000
    return Usage(input_tokens, output_tokens, cost, calls=1)
