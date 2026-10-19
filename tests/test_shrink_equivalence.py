# Equivalence + speed test for the CPU vectorization of shrink_lp_op.
# CPU-only so it runs anywhere (the rest of the suite is CUDA-gated).
# python -m unittest tests.test_shrink_equivalence
import time
import unittest

import torch

import hqq.core.optimize as optimize
from hqq.core.optimize import shrink_lp_op
from hqq.core.quantize import Quantizer


def shrink_lp_op_pow_reference(x, beta, lp_norm):
    # Original implementation: pow(Tensor, non-integer scalar).
    if lp_norm == 1:
        out = torch.abs(x)
        out.sub_(1.0 / beta).clamp_min_(0.0)
        out.mul_(torch.sign(x))
        return out
    out = torch.abs(x)
    out.sub_((1.0 / beta) * out.pow(lp_norm - 1)).clamp_min_(0.0)
    out.mul_(torch.sign(x))
    return out


class TestShrinkEquivalence(unittest.TestCase):
    def test_shrink_op_matches_pow_reference(self):
        torch.manual_seed(0)
        for lp_norm in [0.7, 0.5, 1.0]:
            for beta in [1e1, 1e2, 1e3]:
                x = torch.randn(256, 512, dtype=torch.float32)
                a = shrink_lp_op(x.clone(), beta, lp_norm)
                b = shrink_lp_op_pow_reference(x.clone(), beta, lp_norm)
                self.assertTrue(
                    torch.allclose(a, b, rtol=1e-5, atol=1e-6),
                    f"shrink mismatch at lp_norm={lp_norm}, beta={beta}",
                )

    def test_quantization_output_unchanged(self):
        # End-to-end: the quantized weights must be identical to the pow path.
        torch.manual_seed(0)
        shapes = [(512, 512), (1024, 256), (2048, 128)]
        orig = optimize.shrink_lp_op
        try:
            for shape in shapes:
                W = torch.randn(*shape, dtype=torch.float32)

                optimize.shrink_lp_op = shrink_lp_op_pow_reference
                Quantizer.optimize_weights = optimize.optimize_weights_proximal_legacy
                Wq_ref, meta_ref = Quantizer.quantize(
                    W.clone(), nbits=4, group_size=64, axis=1, device="cpu", bitpack=False
                )

                optimize.shrink_lp_op = orig
                Quantizer.optimize_weights = optimize.optimize_weights_proximal_legacy
                Wq_new, meta_new = Quantizer.quantize(
                    W.clone(), nbits=4, group_size=64, axis=1, device="cpu", bitpack=False
                )

                self.assertTrue(torch.equal(Wq_ref, Wq_new), f"W_q differs for shape {shape}")
                self.assertTrue(torch.allclose(meta_ref["scale"], meta_new["scale"], rtol=1e-6, atol=1e-8))
                self.assertTrue(torch.allclose(meta_ref["zero"], meta_new["zero"], rtol=1e-6, atol=1e-8))
        finally:
            optimize.shrink_lp_op = orig

    def test_shrink_op_is_faster_on_cpu(self):
        # Informational: the vectorized path should not be slower on CPU.
        torch.manual_seed(0)
        x = torch.abs(torch.randn(64, 262144, dtype=torch.float32)) + 1e-3
        beta, lp_norm, n = 1e2, 0.7, 20

        def bench(fn):
            fn(x.clone(), beta, lp_norm)
            t0 = time.time()
            for _ in range(n):
                fn(x.clone(), beta, lp_norm)
            return (time.time() - t0) / n

        t_new = bench(shrink_lp_op)
        t_ref = bench(shrink_lp_op_pow_reference)
        print(f"\nshrink_lp_op CPU: vectorized={t_new*1e3:.2f}ms  pow={t_ref*1e3:.2f}ms  speedup={t_ref/t_new:.2f}x")
        self.assertLessEqual(t_new, t_ref * 1.1)  # allow 10% noise


if __name__ == "__main__":
    unittest.main()
