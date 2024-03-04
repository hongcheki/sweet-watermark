from __future__ import annotations
import collections
from math import sqrt
import pdb
import scipy.stats
from multiprocessing.pool import ThreadPool

import torch
from torch import Tensor
from transformers import (
    LogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)
import pyximport
import sys, os
import numpy as np
import time

include_dirs = [
    os.path.join(sys.path[0],'exp_utils'),
    np.get_include()
]
pyximport.install(
    reload_support=True,
    language_level=sys.version_info[0],
    setup_args={"include_dirs": include_dirs},
)

from exp_utils.levenshtein import levenshtein
from exp_utils import mersenne_rng

class WatermarkBase:
    def __init__(
        self,
        detection_p_threshold: float = 0.1,
        vocab: list[int] = None,
        seeding_scheme: str = "mersenne",  # mostly unused/always default
        hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
        n=256,  # key length
        k=None,
    ):
        # watermarking parameters
        self.detection_p_threshold = detection_p_threshold
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.seeding_scheme = seeding_scheme
        self.key = hash_key
        self.n = n # key length
        self.k = k # block size
        self._seed_rng()
        self._get_secret_key()

    def _seed_rng(self) -> None:
        if self.seeding_scheme == "simple_1":
            self.rng = torch.Generator()
            self.rng.manual_seed(
                self.key
            )  ### newly change self.hash_key to hash_key ###
        elif self.seeding_scheme == "mersenne":
            self.rng = mersenne_rng(self.key)
        else:
            raise NotImplementedError(
                f"Unexpected seeding_scheme: {self.seeding_scheme}"
            )
        return

    def _get_secret_key(self):
        if self.seeding_scheme == "simple_1":
            self.xi = torch.rand(self.n, self.vocab_size, generator=self.rng)
        elif self.seeding_scheme == "mersenne":
            self.xi = torch.tensor(
                [self.rng.rand() for _ in range(self.n * self.vocab_size)]
            ).view(self.n, self.vocab_size)
        print("xi")
        print(self.xi)
        print(self.xi.shape)


class EXPLogitsProcessor(WatermarkBase, LogitsProcessor):
    def __init__(self, *args, **kwargs):
        eos_token_id = kwargs.pop("eos_token_id", None)
        self.eos_token_id = eos_token_id
        self.temperature = kwargs.pop("temperature", 1)
        self.top_p = kwargs.pop("top_p", 1.0)
        self.logits_warper_list = [
            TemperatureLogitsWarper(self.temperature),
            TopPLogitsWarper(self.top_p),
        ]
        super().__init__(*args, **kwargs)

    def _exp_sampling(self, probs, u):
        idx = torch.argmax(u ** (1 / probs), axis=1).unsqueeze(-1)
        return idx

    def preprocess(self, batch_size): # this function must be called before each batch
        self.counter = 0
        self.shifts = torch.randint(self.n, (batch_size,))

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        for logits_warper in self.logits_warper_list:
            scores = logits_warper(input_ids, scores)

        probs = torch.nn.functional.softmax(scores, dim=-1).cpu() # temperature is no use!

        ui = torch.stack(
            [
                self.xi[(self.shifts[i] + self.counter) % self.n, :]
                for i in range(len(input_ids))
            ]
        )
        sampled_index = self._exp_sampling(probs, ui)
        self.counter += 1
        mask = torch.ones(scores.size(), dtype=torch.bool)
        for i, index in enumerate(sampled_index):
            mask[i, index] = False
        scores[mask] = -float("inf")
        return scores


class EXPDetector(WatermarkBase):  # TODO
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def _score_sequence(self, tokens, n, k, vocab_size, n_runs=100):

        def worker(counter):
            xi_alternative = np.random.rand(n, vocab_size).astype(np.float32)
            null_result = self._detect(tokens, n, k, xi_alternative)
            return null_result

        test_result = self._detect(tokens, n, k, np.array(self.xi))

        with ThreadPool(20) as pool:
            null_results = pool.map(worker, range(n_runs))
        p_val = \
            (sum([int(null_result <= test_result) for null_result in null_results]) + 1.0) \
            / \
            (n_runs + 1.0)

        return test_result, null_results, p_val

    def _detect(self, tokens, n, k, xi, gamma=0.0):
        m = len(tokens)
        A = np.empty((m-(k-1), n))
        for i in range(m-(k-1)):
            for j in range(n):
                A[i][j] = levenshtein(tokens[i:i+k],xi[(j+np.arange(k))%n],gamma)

        return np.min(A)

    def detect(
        self,
        generated_tokens,
        **kwargs,
    ) -> dict:
        assert generated_tokens is not None, "Must pass either tokenized string"

        n_runs = kwargs.pop("n_runs", 100)
        output_dict = dict()

        generated_tokens = np.array(generated_tokens)

        if len(generated_tokens) == 0:
            return dict(num_tokens_generated=0, p_value=1.0, true_key_score=None, fake_key_scores=[], prediction=False)

        if self.k is None or self.k > len(generated_tokens):
            k = len(generated_tokens)
        else:
            k = self.k

        test_result, results, p_value = self._score_sequence(
            generated_tokens, self.n, k, self.vocab_size, n_runs
        )
        output_dict.update(dict(num_tokens_generated=len(generated_tokens)))
        output_dict.update(dict(p_value=p_value))
        output_dict.update(dict(true_key_score=test_result))
        output_dict.update(dict(fake_key_scores=results))

        output_dict["prediction"] = p_value < self.detection_p_threshold

        return output_dict
