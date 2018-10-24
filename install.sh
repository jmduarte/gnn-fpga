conda create --copy --name pytorch-training python=3.6
# for gpu:
#conda install --name pytorch-training --file pytorch-training-gpu.conda
# for cpu:
conda install --name pytorch-training --file pytorch-training.conda
source activate pytorch-training
pip install sklearn scipy
git clone https://github.com/LAL/trackml-library
pip install --user trackml-library/
pip install --user uproot
