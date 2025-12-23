# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements **automated circuit tracing with dynamic programming path finding** on top of the `circuit-tracer` library. It extends the attribution graph methodology to find the top-K most influential paths from input embeddings to output logits using exact DP algorithms.

The project consists of:
1. **circuit-tracer submodule**: Library for creating attribution graphs in language models using transcoder features
2. **main.ipynb**: Custom implementation of k-best paths algorithm using optimized DP with topological ordering

## Setup and Installation

### Environment Setup
```bash
# Clone and set up (as per install.sh)
git clone https://github.com/safety-research/circuit-tracer
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install -e circuit-tracer/
uv pip install dotenv
```

### Required Environment Variables
- `HF_TOKEN`: Hugging Face API token (stored in `.env`) for accessing Gemma models and GemmaScope transcoders

### Running the Notebook
```bash
# Activate virtual environment
source .venv/bin/activate

# Launch Jupyter and open main.ipynb
jupyter notebook main.ipynb
```

## Architecture

### Circuit-Tracer Library (`circuit-tracer/`)

The submodule provides core attribution functionality:

- **`circuit_tracer/attribution/`**: Attribution algorithm that computes direct effects between nodes
  - Uses gradient-based methods with frozen linear components to compute edge weights
  - Implements local replacement model to bypass non-linearities

- **`circuit_tracer/graph.py`**: `Graph` class representing attribution graphs
  - Nodes: transcoder features, error nodes, embedding tokens, output logits
  - Adjacency matrix stores direct effects (edge weights) between nodes
  - Node ordering: `[features, errors, embeds, logits]`

- **`circuit_tracer/replacement_model.py`**: `ReplacementModel` wrapper for LLMs with transcoders
  - Integrates pretrained transcoders (GemmaScope for Gemma, etc.)
  - Supports PLTs (per-layer transcoders) and CLTs (cross-layer transcoders)

- **`circuit_tracer/frontend/`**: Visualization server for interactive graph exploration
  - Uses D3.js for rendering attribution graphs
  - Supports node pinning, grouping into supernodes, and annotation

### Main Notebook (`main.ipynb`)

Custom implementation extending circuit-tracer:

**Cell 13-15**: Core DP path-finding algorithm
- `compute_topological_order()`: Kahn's algorithm for DAG ordering
- `find_k_best_paths_dp()`: Optimized k-best paths using DP with lightweight references
  - Processes nodes in reverse topological order
  - Stores only (next_node, edge_weight, score) tuples during DP
  - Reconstructs full paths only at the end (~10-100x faster than naive approach)

**Cell 16**: Finds complete paths from embedding tokens to specific output logits
- Source nodes: Input token embeddings
- Sink node: Target output logit (e.g., "Austin")
- Returns top-K paths ranked by multiplicative influence score

**Cell 18**: Export paths to structured DataFrame for analysis

## Graph Structure

Attribution graphs have 4 node types:

1. **Feature nodes** `[0, n_features)`: Non-zero transcoder features at `(layer, position, feature_idx)`
2. **Error nodes** `[n_features, n_features + n_layers*n_pos)`: Reconstruction errors per layer/position
3. **Embed nodes** `[..., n_features + n_layers*n_pos + n_pos)`: Input token embeddings
4. **Logit nodes** `[..., total_nodes)`: Top-K output logits by probability

**Adjacency matrix**: `adj[target, source]` = direct effect of source→target

## Key Concepts

### Attribution vs Path Finding

- **Attribution** (circuit-tracer): Computes all pairwise direct effects → dense adjacency matrix
- **Path finding** (this repo): Finds specific high-influence paths through the graph using DP

### Influence Score

Paths are ranked by multiplicative score: `∏|edge_weight|` along the path. Higher scores indicate more influential information flow from input to output.

### Topological Ordering

Since attribution graphs are DAGs, processing in reverse topological order ensures all successors are processed before predecessors, enabling efficient DP.

## Development Commands

### Testing (circuit-tracer)
```bash
cd circuit-tracer
pytest                    # Run all tests
pytest tests/test_graph.py  # Run specific test file
```

### Linting and Formatting (circuit-tracer)
```bash
cd circuit-tracer
ruff check               # Lint check
ruff format              # Format code
pyright                  # Type checking
```

### Running Demo Notebooks (circuit-tracer)
```bash
cd circuit-tracer/demos
jupyter notebook circuit_tracing_tutorial.ipynb
```

## Working with Attribution Graphs

### Creating Graphs

```python
from circuit_tracer import ReplacementModel, attribute

# Load model with transcoders
model = ReplacementModel.from_pretrained(
    'google/gemma-2-2b',
    'gemma',  # or HF repo: 'mntss/gemma-scope-transcoders'
    dtype=torch.bfloat16
)

# Run attribution
graph = attribute(
    prompt="Your prompt here",
    model=model,
    max_n_logits=10,
    desired_logit_prob=0.95,
    max_feature_nodes=8192,  # None for unlimited (slower)
    batch_size=256,
    offload='cpu',  # 'disk' if low VRAM, None for GPU-only
    verbose=True
)

# Save graph
graph.to_pt('output.pt')
```

### Finding Paths

```python
# Compute topological order (required for DP)
topo_order = compute_topological_order(graph.adjacency_matrix)

# Define source/sink nodes
n_features = len(graph.selected_features)
n_errors = graph.cfg.n_layers * graph.n_pos
embed_start = n_features + n_errors
embed_nodes = list(range(embed_start, embed_start + graph.n_pos))
target_logit_idx = embed_start + graph.n_pos  # First logit node

# Find top-K paths
paths = find_k_best_paths_dp(
    adj_matrix=graph.adjacency_matrix,
    source_nodes=embed_nodes,
    sink_node=target_logit_idx,
    topo_order=topo_order,
    k=10,
    verbose=True
)
```

### Visualizing Graphs

```python
from circuit_tracer.utils import create_graph_files
from circuit_tracer.frontend.local_server import serve

# Create visualization files
create_graph_files(
    graph_or_path='output.pt',
    slug='my-graph',
    output_path='./graph_files',
    node_threshold=0.8,  # Keep nodes explaining 80% influence
    edge_threshold=0.98  # Keep edges explaining 98% influence
)

# Start server
server = serve(data_dir='./graph_files/', port=8046)
# Visit http://localhost:8046/index.html

# Stop when done
server.stop()
```

## Available Transcoders

- **Gemma-2 (2B)**: `gemma` or `mntss/gemma-scope-transcoders` (PLTs), `mntss/clt-gemma-2-2b-426k` (CLTs)
- **Llama-3.2 (1B)**: `llama` or `mntss/transcoder-Llama-3.2-1B` (PLTs), `mntss/clt-llama-3.2-1b-524k` (CLTs)
- **Qwen-3**: Multiple sizes available (0.6B, 1.7B, 4B, 8B, 14B)

## Important Implementation Details

### Memory Management

The `offload` parameter controls where tensors are stored:
- `None`: Everything on GPU (fastest, high VRAM)
- `'cpu'`: Offload intermediate results to CPU RAM
- `'disk'`: Offload to disk (slowest, lowest VRAM)

### Path Reconstruction Optimization

The optimized DP algorithm stores only `(next_node, edge_weight, score)` tuples during the DP phase, then reconstructs complete paths by following the chain of successors. This avoids expensive list copying and reduces memory usage by ~100x.

### Topological Order Edge Case

If the graph contains cycles (shouldn't happen with attribution graphs), the topological sort will detect this and append remaining nodes to the end with a warning.
