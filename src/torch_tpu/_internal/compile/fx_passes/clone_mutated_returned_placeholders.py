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

"""Clones placeholder nodes that the graph both mutates and returns."""

import operator
from typing import Any

import torch
from torch._functorch._aot_autograd import descriptors as _aot_descriptors
from torch._library import utils as library_utils

# Ops whose schema does not annotate an alias although the result can share the
# input's storage.
_UNANNOTATED_ALIAS_OPS = frozenset({
    torch.ops.aten._unsafe_view.default,  # pylint: disable=protected-access
})


def _normalize_gm_and_graph(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> tuple[torch.fx.GraphModule | None, torch.fx.Graph]:
  """Extracts the owning GraphModule (if any) and the Graph."""
  if isinstance(gm_or_graph, torch.fx.GraphModule):
    return gm_or_graph, gm_or_graph.graph
  return getattr(gm_or_graph, "owning_module", None), gm_or_graph


def _get_output_node(graph: torch.fx.Graph) -> torch.fx.Node | None:
  """Finds and validates that the graph has at most one output node."""
  output_nodes = [node for node in graph.nodes if node.op == "output"]
  if not output_nodes:
    return None

  assert (
      len(output_nodes) == 1
  ), f"Expected exactly one output node, got {len(output_nodes)} output nodes."
  return output_nodes[0]


def _get_nodes(value: Any) -> set[torch.fx.Node]:
  """Extracts all FX nodes in a (possibly nested) argument value."""
  flat_args, _ = torch.utils._pytree.tree_flatten(value)
  return {arg for arg in flat_args if isinstance(arg, torch.fx.Node)}


def _root_descriptor(desc: Any) -> Any:
  """Strips tensor subclass attributes from an AOTAutograd descriptor."""
  while isinstance(
      desc,
      (
          _aot_descriptors.SubclassGetAttrAOTInput,
          _aot_descriptors.SubclassGetAttrAOTOutput,
      ),
  ):
    desc = desc.base
  return desc


def _find_mutated_placeholders(
    graph: torch.fx.Graph, output_node: torch.fx.Node
) -> set[torch.fx.Node]:
  """Returns the placeholders whose storage may be written to.

  An op writes to the arguments annotated `Tensor(a!)` in its schema. A write
  to a view, a `getitem` of a view or an `_UNANNOTATED_ALIAS_OPS` result counts
  as a write to the placeholders it aliases. Targets without a schema are
  assumed to write to all of their inputs and to return aliases of them.

  AOTAutograd graphs return the new value of each input that they mutate, and
  AOTAutograd copies it into the input after the graph runs, so these
  placeholders count as written too.
  """
  mutated_inputs = set()
  for desc in output_node.meta.get("desc") or ():
    desc = _root_descriptor(desc)
    if isinstance(desc, _aot_descriptors.InputMutationAOTOutput):
      mutated_inputs.add(_root_descriptor(desc.mutated_input))

  aliases: dict[torch.fx.Node, set[torch.fx.Node]] = {}
  mutated: set[torch.fx.Node] = set()

  def aliased(value: Any) -> set[torch.fx.Node]:
    return set().union(*(aliases.get(n, ()) for n in _get_nodes(value)))

  for node in graph.nodes:
    if node.op == "placeholder":
      aliases[node] = {node}
      if _root_descriptor(node.meta.get("desc")) in mutated_inputs:
        mutated.add(node)
    elif node.op in ("get_attr", "output"):
      continue
    elif node.op == "call_function" and isinstance(
        node.target, torch._ops.OpOverload  # pylint: disable=protected-access
    ):
      schema = node.target._schema  # pylint: disable=protected-access
      may_alias = set()
      for argument, value in library_utils.zip_schema(
          schema, node.args, node.kwargs
      ):
        if argument.alias_info is not None:
          may_alias |= aliased(value)
          if argument.alias_info.is_write:
            mutated |= aliased(value)
      if node.target in _UNANNOTATED_ALIAS_OPS:
        aliases[node] = aliased(node.all_input_nodes)
      elif any(r.alias_info is not None for r in schema.returns):
        aliases[node] = may_alias
    elif node.op == "call_function" and node.target is operator.getitem:
      aliases[node] = aliased(node.all_input_nodes)
    else:
      aliases[node] = aliased(node.all_input_nodes)
      mutated |= aliases[node]
  return mutated


def _is_returned_compute_placeholder(
    node: torch.fx.Node,
    returned_nodes: set[torch.fx.Node],
) -> bool:
  """Checks if a node is a torch.Tensor output placeholder consumed by compute operations."""
  if node.op != "placeholder" or node not in returned_nodes:
    return False
  if "val" in node.meta and not isinstance(node.meta["val"], torch.Tensor):
    return False
  return any(user.op != "output" for user in node.users)


def _find_placeholders_to_clone(
    graph: torch.fx.Graph,
    output_node: torch.fx.Node,
) -> list[torch.fx.Node]:
  """Identifies all placeholder nodes that need cloning."""
  returned_nodes = _get_nodes(output_node.args)
  mutated = _find_mutated_placeholders(graph, output_node)
  return [
      node
      for node in graph.nodes
      if _is_returned_compute_placeholder(node, returned_nodes)
      and node in mutated
  ]


def _clone_and_replace_placeholder(
    graph: torch.fx.Graph,
    placeholder: torch.fx.Node,
) -> None:
  """Clones a placeholder and redirects all downstream uses to the clone."""
  with graph.inserting_after(placeholder):
    p_clone = graph.call_function(
        torch.ops.aten.clone.default, args=(placeholder,)
    )
    if hasattr(placeholder, "meta"):
      p_clone.meta = placeholder.meta.copy()

    placeholder.replace_all_uses_with(
        p_clone, delete_user_cb=lambda u, clone=p_clone: u != clone
    )


def apply(
    gm_or_graph: torch.fx.GraphModule | torch.fx.Graph,
) -> None:
  """Clones placeholder nodes that are returned in the output and mutated.

  When an FX graph contains in-place operations that mutate input placeholders,
  the graph's output node may reference the input placeholder directly.

  If a pass (e.g. `split_module`) alters the graph structure such that a mutated
  placeholder is no longer in the output, the output will be a passthrough
  meaning that the user won't receive the mutated value.

  A returned placeholder that is only read stays a passthrough instead of
  becoming a program output. AOTAutograd's forward graphs return every
  parameter that the backward graph needs, and cloning them would copy each one
  and keep the copies alive until backward.

  Args:
    gm_or_graph: The FX GraphModule or Graph to modify.
  """
  gm, graph = _normalize_gm_and_graph(gm_or_graph)

  output_node = _get_output_node(graph)
  if output_node is None:
    return

  placeholders_to_clone = _find_placeholders_to_clone(graph, output_node)
  if not placeholders_to_clone:
    return

  for placeholder in placeholders_to_clone:
    _clone_and_replace_placeholder(graph, placeholder)

  graph.lint()
  if gm is not None:
    gm.recompile()
