read -p "Please input the name of environement: " env_name
conda create -n $env_name python=3.9 -y
conda activate $env_name
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt