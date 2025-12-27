git clone https://github.com/safety-research/circuit-tracer
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install -e circuit-tracer/
uv pip install dotenv bs4 hf_transfer matplotlib plotly nbformat
export HF_HUB_ENABLE_HF_TRANSFER=1

curl -fsSL https://claude.ai/install.sh | bash
export IS_SANDBOX=1