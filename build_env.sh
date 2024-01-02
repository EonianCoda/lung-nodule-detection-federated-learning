read -p "Please input the name of environement: " env_name
conda create -n $env_name python=3.9 -y
conda activate $env_name
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install -c conda-forge cudatoolkit=11.2 -y # for cupy-cuda11x
pip install -r requirements.txt