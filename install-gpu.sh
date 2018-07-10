conda create --copy --name pytorch-training python=3.6
conda install --name pytorch-training --file pytorch-training-gpu.conda
source activate pytorch-training
pip install sklearn scipy
