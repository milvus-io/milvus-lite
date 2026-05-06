"""Decay reranker — field-value-based score decay.

Computes a decay factor in (0, 1] based on how far a numeric field
value is from a reference ``origin``.  Three decay curves are supported:

- **gauss**: Gaussian bell curve
- **exp**: Exponential decay
- **linear**: Linear decay (clamps to 0 beyond cutoff)

At ``distance = 0`` (field value equals origin ± offset) the factor is 1.
At ``distance = scale`` the factor equals the ``decay`` parameter.
"""

from __future__ import annotations

import math


class DecayReranker:
    """Local decay reranker (no external API calls).

    Args:
        function: decay curve — ``"gauss"``, ``"exp"``, or ``"linear"``.
        origin: reference value (field values at origin get factor = 1).
        scale: distance at which the factor equals ``decay``.
        offset: "safe zone" around origin where factor stays 1.0.
        decay: target factor at distance = scale (0 < decay < 1).
    """

    _VALID_FUNCTIONS = frozenset({"gauss", "exp", "linear"})

    def __init__(
        self,
        function: str,
        origin: float,
        scale: float,
        offset: float = 0.0,
        decay: float = 0.5,
    ) -> None:
        if function not in self._VALID_FUNCTIONS:
            raise ValueError(
                f"decay: invalid function {function!r}, "
                f"must be one of {sorted(self._VALID_FUNCTIONS)}"
            )
        if scale <= 0:
            raise ValueError(f"decay: scale must be > 0, got {scale}")
        if offset < 0:
            raise ValueError(f"decay: offset must be >= 0, got {offset}")
        if not (0 < decay < 1):
            raise ValueError(
                f"decay: decay must be 0 < decay < 1, got {decay}"
            )

        self._function = function
        self._origin = float(origin)
        self._scale = float(scale)
        self._offset = float(offset)
        self._decay = float(decay)

        # Pre-compute constants used in each curve
        self._ln_decay = math.log(decay)
        if function == "gauss":
            # sigma_sq = scale^2 / ln(decay)  (ln(decay) < 0, so sigma_sq < 0)
            self._sigma_sq = scale * scale / self._ln_decay
        elif function == "linear":
            self._slope = (1.0 - decay) / scale

    def compute_factor(self, field_value: float) -> float:
        """Compute the decay factor for a single field value.

        Returns a value in (0, 1]:
        - 1.0 when ``|field_value - origin| <= offset``
        - ``decay`` when ``|field_value - origin| - offset == scale``
        """
        d = max(0.0, abs(field_value - self._origin) - self._offset)
        if d == 0.0:
            return 1.0

        if self._function == "gauss":
            return math.exp(d * d / self._sigma_sq)
        elif self._function == "exp":
            return math.exp(self._ln_decay * d / self._scale)
        else:  # linear
            return max(0.0, 1.0 - self._slope * d)
