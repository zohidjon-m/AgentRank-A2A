"""
Feature extractors for contextual bandits.

A PayloadFeatureExtractor maps a request payload (e.g. text to be
summarized) into a fixed-length numeric vector. That vector feeds
LinUCB, which learns a per-agent reward function over feature space.

LengthBucketExtractor is intentionally simple — four interpretable
dimensions covering input length. That's enough to capture the
"short tweet vs. long article" distinction that motivates contextual
ranking in the first place. New extractors can be added by registering
them in FEATURE_EXTRACTORS.
"""

from abc import ABC, abstractmethod
from math import log
from typing import Dict

import numpy as np


class PayloadFeatureExtractor(ABC):
    name: str = "abstract"

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def extract(self, payload: str) -> np.ndarray:
        ...


class LengthBucketExtractor(PayloadFeatureExtractor):
    """
    4-dimensional feature vector:

      [0] = 1.0                              intercept
      [1] = log(1+words) / log(1000)         normalized log word count
      [2] = 1.0 if words < 30 else 0.0       short indicator
      [3] = 1.0 if words > 200 else 0.0      long indicator

    Bounded to [0, 1] so LinUCB confidence terms stay well-scaled.
    """
    name = "length_bucket"

    @property
    def dim(self) -> int:
        return 4

    def extract(self, payload: str) -> np.ndarray:
        words = (payload or "").split()
        wc = len(words)
        norm_log = log(1 + wc) / log(1000)
        return np.array(
            [
                1.0,
                min(1.0, norm_log),
                1.0 if wc < 30 else 0.0,
                1.0 if wc > 200 else 0.0,
            ],
            dtype=float,
        )


# Registry of known extractors. Keys must match config values.
FEATURE_EXTRACTORS: Dict[str, PayloadFeatureExtractor] = {
    "length_bucket": LengthBucketExtractor(),
}


def get_extractor(name: str) -> PayloadFeatureExtractor:
    if name not in FEATURE_EXTRACTORS:
        raise KeyError(
            f"Unknown feature extractor {name!r}. "
            f"Known: {sorted(FEATURE_EXTRACTORS.keys())}"
        )
    return FEATURE_EXTRACTORS[name]
