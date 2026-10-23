# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for splash_sdpa: grouped K/V heads and block sizes.

Outputs and gradients are compared against a float32 attention computed on the
CPU with an explicit mask. TPU matmuls round float32 operands to bfloat16 by
default, so those comparisons allow an error relative to the largest reference
value (see `_assert_all_close`).
"""

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch_tpu._internal.ops import splash_attention
from torch_tpu._internal.utils import test_utils
from tests import seed_test_utils

_REL_TOL = 2e-2
_BF16_REL_TOL = 3e-2


def _reference_attention(q, k, v, *, is_causal=True, local_window_size=None):
  """float32 attention with an explicit mask; repeats grouped K/V heads."""
  seq_len = q.shape[2]
  i = torch.arange(seq_len)[:, None]
  j = torch.arange(seq_len)[None, :]
  if local_window_size is not None:
    allowed = (j <= i) & (i - j <= local_window_size)
  elif is_causal:
    allowed = j <= i
  else:
    allowed = torch.ones(seq_len, seq_len, dtype=torch.bool)
  allowed = allowed.expand(q.shape[0], seq_len, seq_len)
  group = q.shape[1] // k.shape[1]
  k = k.repeat_interleave(group, dim=1)
  v = v.repeat_interleave(group, dim=1)
  scores = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5
  scores = scores.masked_fill(~allowed[:, None], float("-inf"))
  return torch.softmax(scores, dim=-1) @ v


class SplashAttentionTest(seed_test_utils.RepeatableTest):

  def setUp(self):
    super().setUp()
    self.device = torch.device("tpu")

  def _inputs(
      self, n_heads, n_kv_heads, seq_len=1024, batch=2, dtype=torch.float32
  ):
    """Returns CPU tensors (q, k, v, grad_out) in (B, H, S, D) layout."""
    head_dim = 128
    q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype)
    k = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype)
    v = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype)
    grad_out = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype)
    return q, k, v, grad_out

  def _run_splash(self, q, k, v, grad_out, **kwargs):
    """Runs splash_sdpa on the TPU; returns CPU (out, dq, dk, dv)."""
    q, k, v = (
        t.detach().to(self.device).requires_grad_(True) for t in (q, k, v)
    )
    out = splash_attention.splash_sdpa(
        q, k, v, enable_gqa=q.shape[1] != k.shape[1], **kwargs
    )
    out.backward(grad_out.to(self.device))
    return tuple(t.detach().cpu() for t in (out, q.grad, k.grad, v.grad))

  def _run_reference(self, q, k, v, grad_out, **kwargs):
    """Runs the float32 CPU reference; returns (out, dq, dk, dv)."""
    q, k, v = (t.detach().float().requires_grad_(True) for t in (q, k, v))
    out = _reference_attention(q, k, v, **kwargs)
    out.backward(grad_out.float())
    return tuple(t.detach() for t in (out, q.grad, k.grad, v.grad))

  def _assert_all_close(self, actual, expected, rel):
    """Compares (out, dq, dk, dv) with atol scaled by the largest |expected|."""
    for name, a, e in zip(("out", "dq", "dk", "dv"), actual, expected):
      test_utils.assert_close(
          a.float(),
          e.float(),
          rtol=rel,
          atol=rel * e.abs().max().item(),
          preamble=name,
      )

  @parameterized.product(
      is_causal=(True, False),
      heads=((8, 8), (8, 2), (8, 1)),
  )
  def test_matches_reference(self, is_causal, heads):
    """MHA, GQA (grouped K/V read natively) and MQA against the reference."""
    n_heads, n_kv_heads = heads
    q, k, v, grad_out = self._inputs(n_heads, n_kv_heads)
    actual = self._run_splash(q, k, v, grad_out, is_causal=is_causal)
    expected = self._run_reference(q, k, v, grad_out, is_causal=is_causal)
    self._assert_all_close(actual, expected, _REL_TOL)

  @parameterized.named_parameters(
      dict(testcase_name="local_window", local_window_size=256),
      dict(testcase_name="unfused_bwd", use_fused_bwd_kernel=False),
      dict(testcase_name="bfloat16", dtype=torch.bfloat16),
  )
  def test_grouped_kv_matches_reference(
      self,
      local_window_size=None,
      use_fused_bwd_kernel=True,
      dtype=torch.float32,
  ):
    q, k, v, grad_out = self._inputs(8, 2, dtype=dtype)
    actual = self._run_splash(
        q,
        k,
        v,
        grad_out,
        local_window_size=local_window_size,
        use_fused_bwd_kernel=use_fused_bwd_kernel,
    )
    expected = self._run_reference(
        q, k, v, grad_out, local_window_size=local_window_size
    )
    rel = _BF16_REL_TOL if dtype == torch.bfloat16 else _REL_TOL
    self._assert_all_close(actual, expected, rel)

  @parameterized.parameters(768, 1280)
  def test_default_blocks_for_lengths_not_divisible_by_512(self, seq_len):
    q, k, v, grad_out = self._inputs(8, 2, seq_len=seq_len, batch=1)
    actual = self._run_splash(q, k, v, grad_out)
    expected = self._run_reference(q, k, v, grad_out)
    self._assert_all_close(actual, expected, _REL_TOL)

  @parameterized.named_parameters(
      ("block_kv", dict(block_kv=256)),
      ("block_dkv", dict(block_dkv=256)),
  )
  def test_explicit_blocks_smaller_than_512(self, block_sizes):
    """The compute block defaults divide an explicit enclosing block."""
    q, k, v, grad_out = self._inputs(8, 2)
    actual = self._run_splash(q, k, v, grad_out, **block_sizes)
    expected = self._run_reference(q, k, v, grad_out)
    self._assert_all_close(actual, expected, _REL_TOL)

  def test_rejects_kv_heads_that_do_not_divide_q_heads(self):
    q, k, v, _ = self._inputs(6, 4, batch=1)
    q, k, v = (t.to(self.device) for t in (q, k, v))
    with self.assertRaisesRegex(ValueError, "must divide"):
      splash_attention.splash_sdpa(q, k, v, enable_gqa=True)

  def test_rejects_lengths_not_multiple_of_128(self):
    q, k, v, _ = self._inputs(8, 8, seq_len=200, batch=1)
    q, k, v = (t.to(self.device) for t in (q, k, v))
    with self.assertRaisesRegex(ValueError, "cannot pick a default block size"):
      splash_attention.splash_sdpa(q, k, v)


class DefaultBlockSizeTest(seed_test_utils.RepeatableTest):

  @parameterized.parameters(
      (128, 128),
      (384, 384),
      (512, 512),
      (640, 128),
      (768, 384),
      (1024, 512),
      (1280, 256),
      (1920, 384),
      (2560, 512),
      (8192, 512),
  )
  def test_default_block_size(self, length, expected):
    self.assertEqual(splash_attention._default_block_size(length), expected)

  @parameterized.parameters(0, 64, 200, 1000)
  def test_default_block_size_rejects_unaligned_lengths(self, length):
    with self.assertRaisesRegex(ValueError, "cannot pick a default block size"):
      splash_attention._default_block_size(length)


if __name__ == "__main__":
  absltest.main()
