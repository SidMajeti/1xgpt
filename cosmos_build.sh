python3 -m venv cosmos
source cosmos/bin/activate && pip install av && git clone https://github.com/NVIDIA/Cosmos-Tokenizer.git &&
cd Cosmos-Tokenizer && pip install -e .
python3 download_cosmos.py