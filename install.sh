git clone https://github.com/safety-research/circuit-tracer
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
uv venv
source .venv/bin/activate
uv pip install -e circuit-tracer/
uv pip install dotenv bs4