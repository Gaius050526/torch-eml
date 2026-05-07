"""Visualization for EML trees."""

from __future__ import annotations

from pathlib import Path

from torch_eml.tree import EMLTree
from torch_eml.pruning import ConstantNode


def _node_info(node, idx: int) -> dict:
    """Extract display info from a node."""
    is_pruned = isinstance(node, ConstantNode)
    if is_pruned:
        return {
            "idx": idx,
            "pruned": True,
            "label": f"{node.value:.2f}",
            "w_left": 0.0,
            "w_right": 0.0,
            "bias_left": 0.0,
            "bias_right": 0.0,
        }
    return {
        "idx": idx,
        "pruned": False,
        "label": "eml",
        "w_left": node.w_left.item(),
        "w_right": node.w_right.item(),
        "bias_left": node.bias_left.item(),
        "bias_right": node.bias_right.item(),
    }


def _build_layout(tree: EMLTree) -> list[dict]:
    """Assign (x, y) positions to each node in the tree using BFS level-order."""
    nodes_info = []
    width = 2**tree.depth * 60
    y_spacing = 80

    # Process level by level (top-down)
    # Node 0 is root, children of node i are at 2i+1 and 2i+2
    for idx in range(len(tree.nodes)):
        # Find level and position within level
        level = 0
        start = 0
        while start + 2**level <= idx:
            start += 2**level
            level += 1
        pos_in_level = idx - start
        n_at_level = 2**level

        x = width * (pos_in_level + 0.5) / n_at_level
        y = 40 + level * y_spacing

        info = _node_info(tree.nodes[idx], idx)
        info["x"] = x
        info["y"] = y
        info["level"] = level
        nodes_info.append(info)

    return nodes_info


def tree_to_html(
    tree_or_head,
    title: str = "EML Tree",
    equation: str | None = None,
) -> str:
    """Generate standalone HTML visualization of an EML tree.

    Args:
        tree_or_head: An EMLTree or EMLHead instance.
        title: Page title.
        equation: Optional equation string to display below the tree.

    Returns:
        Complete HTML string.
    """
    tree = tree_or_head.tree if hasattr(tree_or_head, "tree") else tree_or_head

    nodes = _build_layout(tree)
    width = 2**tree.depth * 60
    height = (tree.depth + 1) * 80 + 60

    # Build SVG edges
    edges_svg = []
    for info in nodes:
        idx = info["idx"]
        left_child = 2 * idx + 1
        right_child = 2 * idx + 2
        if left_child < len(nodes):
            child = nodes[left_child]
            color = "#ccc" if child["pruned"] else "#4a9eff"
            edges_svg.append(
                f'<line x1="{info["x"]}" y1="{info["y"]}" '
                f'x2="{child["x"]}" y2="{child["y"]}" '
                f'stroke="{color}" stroke-width="2" />'
            )
        if right_child < len(nodes):
            child = nodes[right_child]
            color = "#ccc" if child["pruned"] else "#ff6b4a"
            edges_svg.append(
                f'<line x1="{info["x"]}" y1="{info["y"]}" '
                f'x2="{child["x"]}" y2="{child["y"]}" '
                f'stroke="{color}" stroke-width="2" />'
            )

    # Build SVG nodes
    nodes_svg = []
    for info in nodes:
        if info["pruned"]:
            fill = "#f0f0f0"
            stroke = "#ccc"
            text_color = "#999"
        else:
            fill = "#1a1a2e"
            stroke = "#4a9eff"
            text_color = "#fff"

        tooltip = (
            f"Node {info['idx']}\\n"
            f"w_left={info['w_left']:.4f}, bias_left={info['bias_left']:.4f}\\n"
            f"w_right={info['w_right']:.4f}, bias_right={info['bias_right']:.4f}"
        )
        if info["pruned"]:
            tooltip = f"Node {info['idx']} (pruned)\\nConstant: {info['label']}"

        nodes_svg.append(
            f'<g class="node" onmouseover="showTooltip(evt, \'{tooltip}\')" '
            f'onmouseout="hideTooltip()">'
            f'<circle cx="{info["x"]}" cy="{info["y"]}" r="22" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2" />'
            f'<text x="{info["x"]}" y="{info["y"] + 5}" '
            f'text-anchor="middle" fill="{text_color}" '
            f'font-family="monospace" font-size="11">{info["label"]}</text>'
            f"</g>"
        )

    # Leaf labels
    leaves_svg = []
    n_leaves = tree.n_leaves
    for i in range(n_leaves):
        x = width * (i + 0.5) / n_leaves
        y_pos = height - 15
        leaves_svg.append(
            f'<text x="{x}" y="{y_pos}" text-anchor="middle" '
            f'fill="#888" font-family="monospace" font-size="10">x{i}</text>'
        )

    eq_html = ""
    if equation:
        eq_html = (
            f'<div style="margin-top:16px;padding:12px 20px;background:#1a1a2e;'
            f'border-radius:8px;font-family:monospace;color:#4a9eff;'
            f'font-size:14px;word-break:break-all;">f(x) = {equation}</div>'
        )

    html_str = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{
    background: #0d0d1a;
    color: #eee;
    font-family: -apple-system, sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 30px;
    margin: 0;
  }}
  h1 {{
    font-size: 24px;
    color: #4a9eff;
    margin-bottom: 8px;
  }}
  .subtitle {{
    color: #888;
    font-size: 14px;
    margin-bottom: 24px;
  }}
  .node circle {{
    cursor: pointer;
    transition: all 0.2s;
  }}
  .node:hover circle {{
    stroke-width: 3;
    filter: brightness(1.3);
  }}
  #tooltip {{
    position: fixed;
    background: #16213e;
    border: 1px solid #4a9eff;
    border-radius: 6px;
    padding: 8px 12px;
    font-family: monospace;
    font-size: 12px;
    color: #eee;
    pointer-events: none;
    white-space: pre;
    display: none;
    z-index: 100;
  }}
  .legend {{
    display: flex;
    gap: 20px;
    margin-top: 16px;
    font-size: 13px;
    color: #888;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .legend-dot {{
    width: 12px;
    height: 12px;
    border-radius: 50%;
  }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="subtitle">depth={tree.depth}, nodes={len(tree.nodes)}, leaves={tree.n_leaves}</div>
<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  {''.join(edges_svg)}
  {''.join(nodes_svg)}
  {''.join(leaves_svg)}
</svg>
<div class="legend">
  <div class="legend-item">
    <div class="legend-dot" style="background:#4a9eff"></div> left (exp)
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#ff6b4a"></div> right (ln)
  </div>
  <div class="legend-item">
    <div class="legend-dot" style="background:#ccc"></div> pruned
  </div>
</div>
{eq_html}
<div id="tooltip"></div>
<script>
function showTooltip(evt, text) {{
  const t = document.getElementById('tooltip');
  t.textContent = text;
  t.style.display = 'block';
  t.style.left = (evt.clientX + 15) + 'px';
  t.style.top = (evt.clientY - 10) + 'px';
}}
function hideTooltip() {{
  document.getElementById('tooltip').style.display = 'none';
}}
</script>
</body>
</html>"""
    return html_str


def save_html(
    tree_or_head,
    path: str,
    title: str = "EML Tree",
    equation: str | None = None,
) -> str:
    """Generate and save HTML visualization to a file.

    Args:
        tree_or_head: An EMLTree or EMLHead instance.
        path: Output file path.
        title: Page title.
        equation: Optional equation string to display.

    Returns:
        Absolute path to the saved file.
    """
    html_str = tree_to_html(tree_or_head, title=title, equation=equation)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_str)
    return str(p.resolve())
