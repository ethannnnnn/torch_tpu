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

"""Tests for clone_mutated_returned_placeholders FX pass."""

import copy
import operator

from absl.testing import absltest
import torch
from torch import fx
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.testing._internal.two_tensor import TwoTensor
from torch_tpu._internal.compile.fx_passes import clone_mutated_returned_placeholders
from torch_tpu._internal.utils import test_utils as utils
from tests import seed_test_utils

GraphModule = fx.GraphModule
Graph = fx.Graph
aten = torch.ops.aten


def _cloned_placeholders(graph: Graph) -> set[str]:
  return {
      n.args[0].name
      for n in graph.nodes
      if n.op == "call_function" and n.target == aten.clone.default
  }


class _NormalizeByRunningVariance(torch.nn.Module):
  """Reads a buffer in the forward and then updates it in place."""

  def __init__(self, running_var: torch.Tensor):
    super().__init__()
    self.linear = torch.nn.Linear(16, 32, bias=False)
    self.register_buffer("running_var", running_var)

  def forward(self, x):
    h = self.linear(x)
    y = h * torch.rsqrt(self.running_var + 1e-5)
    with torch.no_grad():
      self.running_var.lerp_(h.var(0), 0.1)
    return y


class CloneMutatedReturnedPlaceholdersTest(seed_test_utils.RepeatableTest):

  def test_clones_mutated_returned_placeholder(self):
    """Verifies that mutated placeholders returned in the output are cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place addition mutating p_weight
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    # The output node returns the mutated placeholder directly
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    # Check that a clone node was created right after p_weight
    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    clone_node = clone_nodes[0]
    self.assertEqual(clone_node.args, (p_weight,))
    self.assertEqual(clone_node.meta["val"].shape, p_weight.meta["val"].shape)

    # p_weight's only user should be the clone node
    self.assertEqual(list(p_weight.users.keys()), [clone_node])

    # Downstream compute and output should now use the clone node
    self.assertIn(clone_node, add_node.args)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    self.assertIn(clone_node, output_node.args[0])
    self.assertNotIn(p_weight, output_node.args[0])

    # Verify that the generated code was recompiled to invoke clone
    self.assertIn("clone", gm.code)

  def test_apply_on_graph_directly(self):
    """Verifies that apply() works directly on a raw torch.fx.Graph instance."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))

    # Apply directly on graph, where gm is None
    clone_mutated_returned_placeholders.apply(graph)

    clone_nodes = [
        n
        for n in graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 1)
    self.assertEqual(list(p_weight.users.keys()), [clone_nodes[0]])

  def test_recompiles_graph_module(self):
    """Verifies that apply() triggers gm.recompile(), updating gm.code."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_grad = graph.placeholder("grad")
    add_node = graph.call_function(
        torch.ops.aten.add_.Tensor, (p_weight, p_grad)
    )
    graph.output((add_node, p_weight))
    gm = GraphModule(torch.nn.Module(), graph)

    self.assertNotIn("clone", gm.code)
    clone_mutated_returned_placeholders.apply(gm)
    self.assertIn("clone", gm.code)

  def test_passthrough_placeholder_not_cloned(self):
    """Verifies that placeholders only returned without compute uses are not cloned."""
    graph = Graph()
    p_in = graph.placeholder("in_tensor")
    p_in.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    graph.output((p_in,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_compute_placeholder_not_returned_not_cloned(self):
    """Verifies that compute placeholders not present in the output are not cloned."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_x.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_y = graph.placeholder("y")
    p_y.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    out = graph.call_function(torch.ops.aten.add.Tensor, (p_x, p_y))
    graph.output((out,))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_multiple_mutated_returned_placeholders(self):
    """Verifies multiple placeholders are all cloned."""
    graph = Graph()
    p_weight = graph.placeholder("weight")
    p_weight.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_m = graph.placeholder("exp_avg")
    p_m.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_v = graph.placeholder("exp_avg_sq")
    p_v.meta["val"] = torch.empty((4, 4), dtype=torch.float32)
    p_step = graph.placeholder("step")
    p_step.meta["val"] = torch.empty((), dtype=torch.int32)
    p_grad = graph.placeholder("grad")
    p_grad.meta["val"] = torch.empty((4, 4), dtype=torch.float32)

    # In-place ops mutating weight, m, v, step
    graph.call_function(torch.ops.aten.add_.Tensor, (p_step, 1))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_m, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_v, p_grad))
    graph.call_function(torch.ops.aten.add_.Tensor, (p_weight, p_m))

    graph.output((p_weight, p_m, p_v, p_step))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertLen(clone_nodes, 4)
    for p in (p_weight, p_m, p_v, p_step):
      self.assertLen(p.users, 1)
      cloned_user = list(p.users.keys())[0]
      self.assertEqual(cloned_user.target, torch.ops.aten.clone.default)

  def test_symint_placeholder_not_cloned(self):
    """Verifies that SymInt placeholders in compute/output are not cloned."""
    shape_env = torch.fx.experimental.symbolic_shapes.ShapeEnv()
    symint_s0 = shape_env.create_symintnode(
        shape_env.create_symbol(4, torch._dynamo.source.ConstantSource("s0")),
        hint=4,
    )

    fake_mode = torch._subclasses.fake_tensor.FakeTensorMode(
        shape_env=shape_env
    )
    with fake_mode:
      fake_tensor = torch.empty((symint_s0, 4), dtype=torch.float32)

    graph = Graph()
    p_s0 = graph.placeholder("s0")
    p_s0.meta["val"] = symint_s0
    p_x = graph.placeholder("x")
    p_x.meta["val"] = fake_tensor

    # p_s0 is used in compute (slice) and also returned in output
    slice_node = graph.call_function(
        torch.ops.aten.slice.Tensor, (p_x, 0, 0, p_s0)
    )
    graph.output((slice_node, p_s0))
    gm = GraphModule(torch.nn.Module(), graph)

    clone_mutated_returned_placeholders.apply(gm)

    clone_nodes = [
        n
        for n in gm.graph.nodes
        if n.op == "call_function" and n.target == torch.ops.aten.clone.default
    ]
    self.assertEmpty(clone_nodes)

  def test_read_only_returned_placeholder_not_cloned(self):
    """Verifies that a returned placeholder that is only read is not cloned."""
    graph = Graph()
    p_x = graph.placeholder("x")
    p_weight = graph.placeholder("weight")
    t = graph.call_function(aten.t.default, (p_weight,))
    graph.output((graph.call_function(aten.mm.default, (p_x, t)), p_weight))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEmpty(_cloned_placeholders(graph))

  def test_placeholder_written_through_alias_is_cloned(self):
    """Verifies writes through views, getitem, out= and schema-less targets."""
    graph = Graph()
    p_view = graph.placeholder("view")
    p_split = graph.placeholder("split")
    p_unsafe = graph.placeholder("unsafe")
    p_out = graph.placeholder("out")
    p_opaque = graph.placeholder("opaque")
    p_read = graph.placeholder("read")
    graph.call_function(
        aten.add_.Scalar,
        (graph.call_function(aten.view.default, (p_view, [16])), 1.0),
    )
    pieces = graph.call_function(aten.split.Tensor, (p_split, 2))
    graph.call_function(
        aten.zero_.default,
        (graph.call_function(operator.getitem, (pieces, 0)),),
    )
    graph.call_function(
        aten.add_.Scalar,
        (graph.call_function(aten._unsafe_view.default, (p_unsafe, [16])), 1.0),
    )
    out = graph.call_function(aten.add.out, (p_read, p_read), {"out": p_out})
    graph.call_function(aten.add_.Scalar, (out, 1.0))
    graph.call_function(lambda t: t, (p_opaque,))
    read_pieces = graph.call_function(aten.split.Tensor, (p_read, 2))
    graph.call_function(operator.getitem, (read_pieces, 0))
    fresh = graph.call_function(aten.mul.Scalar, (p_read, 2.0))
    graph.call_function(
        aten.add_.Scalar,
        (graph.call_function(aten.view.default, (fresh, [16])), 1.0),
    )
    graph.output((p_view, p_split, p_unsafe, p_out, p_opaque, p_read))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(
        _cloned_placeholders(graph),
        {"view", "split", "unsafe", "out", "opaque"},
    )

  def test_placeholder_written_through_unannotated_alias_is_cloned(self):
    """Verifies writes through aliases that op schemas do not annotate."""
    graph = Graph()
    names = (
        "unsafe_split",
        "unsafe_split_sizes",
        "unsafe_chunk",
        "lift",
        "functional_set",
        "source",
        "offset_source",
        "rebound",
        "new_source",
        "read",
    )
    p = {name: graph.placeholder(name) for name in names}

    def write(node):
      graph.call_function(aten.add_.Scalar, (node, 1.0))

    def empty():
      return graph.call_function(aten.empty.memory_format, ([0],))

    for name, target, size in (
        ("unsafe_split", aten.unsafe_split.Tensor, 2),
        ("unsafe_split_sizes", aten.unsafe_split_with_sizes.default, [2, 2]),
        ("unsafe_chunk", aten.unsafe_chunk.default, 2),
    ):
      pieces = graph.call_function(target, (p[name], size))
      write(graph.call_function(operator.getitem, (pieces, 1)))
    write(graph.call_function(aten.lift.default, (p["lift"],)))
    for name, target in (
        ("functional_set", aten.set.source_Tensor),
        ("source", aten.set_.source_Tensor),
    ):
      write(graph.call_function(target, (empty(), p[name])))
    offset_view = empty()
    graph.call_function(
        aten.set_.source_Tensor_storage_offset,
        (offset_view, p["offset_source"], 0, [16], [1]),
    )
    write(offset_view)
    graph.call_function(
        aten.set_.source_Tensor, (p["rebound"], p["new_source"])
    )
    write(p["rebound"])
    graph.call_function(aten.set_.source_Tensor, (empty(), p["read"]))
    graph.output(tuple(p.values()))

    clone_mutated_returned_placeholders.apply(graph)

    self.assertEqual(_cloned_placeholders(graph), set(names) - {"read"})

  def _check_compiled_forward_clones_only_running_var(
      self, running_var: torch.Tensor
  ):
    """Compiles `_NormalizeByRunningVariance` with TpuBackend's settings.

    The forward graph returns the weight and the input for backward, and the
    buffer too, because min-cut recomputes `rsqrt` in backward. Only the
    buffer, which AOTAutograd updates after the graph runs, should be cloned.

    Args:
      running_var: The buffer, of shape (32,).
    """
    forward_graphs = []

    def fw_compiler(graph_module, example_inputs):
      del example_inputs
      clone_mutated_returned_placeholders.apply(graph_module)
      forward_graphs.append(graph_module)
      return graph_module

    torch._dynamo.reset()
    model = _NormalizeByRunningVariance(running_var)
    eager_model = copy.deepcopy(model)
    compiled = torch.compile(
        model,
        backend=aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=lambda graph_module, _: graph_module,
            partition_fn=min_cut_rematerialization_partition,
            keep_inference_input_mutations=False,
        ),
        fullgraph=True,
    )
    x = torch.randn(8, 16, requires_grad=True)
    eager_x = x.detach().clone().requires_grad_()

    compiled(x).sum().backward()
    eager_model(eager_x).sum().backward()

    (graph_module,) = forward_graphs
    cloned = [
        n.args[0].meta["val"].shape
        for n in graph_module.graph.nodes
        if n.op == "call_function" and n.target == aten.clone.default
    ]
    self.assertEqual(set(cloned), {running_var.shape})
    utils.assert_close(x.grad, eager_x.grad)
    utils.assert_close(model.linear.weight.grad, eager_model.linear.weight.grad)
    utils.assert_close(model.running_var, eager_model.running_var)

  def test_compiled_forward_clones_only_the_input_it_mutates(self):
    """Verifies that only the input that AOTAutograd mutates is cloned."""
    self._check_compiled_forward_clones_only_running_var(torch.ones(32))

  def test_compiled_forward_clones_mutated_tensor_subclass_input(self):
    """Verifies that a mutated subclass input's inner tensors are cloned."""
    self._check_compiled_forward_clones_only_running_var(
        TwoTensor(torch.ones(32), torch.ones(32))
    )


if __name__ == "__main__":
  absltest.main()
