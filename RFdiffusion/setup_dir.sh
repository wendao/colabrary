apt-get install aria2
mkdir params
# send param download into background
aria2c -q -x 16 https://files.ipd.uw.edu/krypton/schedules.zip
aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
aria2c -q -x 16 http://files.ipd.uw.edu/pub/RFdiffusion/f572d396fae9206628714fb2ce00f72e/Complex_beta_ckpt.pt
aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar -C params
touch params/done.txt

#git clone https://github.com/sokrypton/RFdiffusion.git
#pip -q install jedi omegaconf hydra-core icecream pyrsistent
#pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
#cd RFdiffusion/env/SE3Transformer
#pip -q install --no-cache-dir -r requirements.txt
#pip -q install .

wget -qnc https://files.ipd.uw.edu/krypton/ananas
chmod +x ananas
