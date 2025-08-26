import math
from typing import Callable

import numpy as np
import torch

# We are using int64, but treating it as unsigned.
# We cannot use uint64 because torch doesn't support bitwise shift for it.
CODEWORD_LENGTH = 64


# Returns 64-bit number with bit in position `bit_id` equal to 1 and all other bits equal to 0.
def _one_shifted(bit_id: int) -> int:
    return -0x8000000000000000 if bit_id == 63 else (1 << bit_id)


# Returns 64-bit number with `n` highest bits set to 0, and all other bits set to 1.
# This must be applied after shift right by n in case the sign bit might have been set.
def _mask_with_high_zeros(n: int) -> int:
    return (1 << (64 - n)) - 1


class StringEncoder:
    """Helper class to encode strings that represent elements of coset.

    Original (decoded) strings are 2D tensors where tensor elements are integers representing elements being permuted.
    In encoded format, these elements are compressed to take less memory. Each element takes only `code_width` bits.
    For binary strings (`code_width=1`) and `n<=63`, this allows to represent coset element with a single int64 number.
    Elements in the original string must be in range `[0, 2**code_width)`.
    This class also provides functionality to efficiently apply permutation in encoded format using bit operations.
    """

    def __init__(self, *, code_width: int = 1, n: int = 1):
        """Initializes StringEncoder.

        :param code_width: Number of bits to encode one element of coset.
        :param n: Length of the string to encode.
        """
        assert 1 <= code_width <= CODEWORD_LENGTH
        self.w = code_width
        self.n = n
        self.uses_sign_bit = (self.w * self.n) >= 64
        self.encoded_length = int(math.ceil(self.n * self.w / CODEWORD_LENGTH))  # Encoded length.

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        """Encodes tensor of coset elements.

        Input shape `(m, self.n)`. Output shape `(m, self.encoded_length)`.
        """
        assert len(s.shape) == 2
        assert s.shape[1] == self.n
        assert torch.min(s) >= 0, "Cannot encode negative values."
        max_value = torch.max(s)
        assert max_value < 2**self.w, f"Width {self.w} is not sufficient to encode value {max_value}."

        encoded = torch.zeros((s.shape[0], self.encoded_length), dtype=torch.int64, device=s.device)
        w, cl = self.w, CODEWORD_LENGTH
        for i in range(w * self.n):
            encoded[:, i // cl] |= ((s[:, i // w] >> (i % w)) & 1) << (i % cl)
        return encoded

    def decode(self, encoded: torch.Tensor) -> torch.Tensor:
        """Decodes tensor of coset elements.

        Input shape `(m, self.encoded_length)`. Output shape `(m, self.n)`.
        """
        orig = torch.zeros((encoded.shape[0], self.n), dtype=torch.int64, device=encoded.device)
        w, cl = self.w, CODEWORD_LENGTH
        for i in range(w * self.n):
            orig[:, i // w] |= ((encoded[:, i // cl] >> (i % cl)) & 1) << (i % w)
        return orig

    def prepare_shift_to_mask(self, p: list[int]) -> dict[tuple[int, int, int], int]:
        assert len(p) == self.n
        shift_to_mask: dict[tuple[int, int, int], int] = {}
        for i in range(self.n):
            for j in range(self.w):
                start_bit = int(p[i] * self.w + j)
                end_bit = i * self.w + j
                start_cw_id = start_bit // CODEWORD_LENGTH
                end_cw_id = end_bit // CODEWORD_LENGTH
                shift = (end_bit % CODEWORD_LENGTH) - (start_bit % CODEWORD_LENGTH)
                key = (start_cw_id, end_cw_id, shift)
                if key not in shift_to_mask:
                    shift_to_mask[key] = 0
                shift_to_mask[key] |= _one_shifted(start_bit % CODEWORD_LENGTH)
        return shift_to_mask

    def implement_permutation(self, p: list[int]) -> Callable[[torch.Tensor, torch.Tensor], None]:
        """Converts permutation to a function on encoded tensor implementing this permutation.

        This function writes result to tensor in second argument, which must be initialized to zeros.
        """
        shift_to_mask = self.prepare_shift_to_mask(p)
        lines = ["def f_(x,y):"]
        for (start_cw_id, end_cw_id, shift), mask in shift_to_mask.items():
            line = f" y[:,{end_cw_id}] |= (x[:,{start_cw_id}] & {mask})"
            if shift > 0:
                line += f"<<{shift}"
            elif shift < 0:
                line += f">>{-shift}"
                if mask < 0:
                    line += f"&{_mask_with_high_zeros(-shift)}"
            lines.append(line)
        src = "\n".join(lines)
        l: dict = {}
        exec(src, {}, l)  # pylint: disable=exec-used
        return l["f_"]

    def implement_permutation_1d(self, p: list[int]) -> Callable[[np.ndarray], np.ndarray]:
        """Converts permutation to a function on encoded tensor implementing this permutation.

        The function converts 1D tensor to 1D tensor of the same dimension.
        Applicable only if state can be encoded by single int64 (encoded_length=1).
        """
        assert self.encoded_length == 1
        shift_to_mask = self.prepare_shift_to_mask(p)
        terms = []
        for (_, _, shift), mask in shift_to_mask.items():
            term = f"(x&{mask})"
            if shift > 0:
                term += f"<<{shift}"
            elif shift < 0:
                term += f">>{-shift}"
                if mask < 0:
                    term += f"&{_mask_with_high_zeros(-shift)}"
            terms.append(f"({term})")
        src = "f_ = lambda x: " + " | ".join(terms)
        l: dict = {}
        exec(src, {}, l)  # pylint: disable=exec-used
        return l["f_"]
