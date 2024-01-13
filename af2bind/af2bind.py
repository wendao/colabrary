import os
from colabdesign import mk_afdesign_model, clear_mem
import numpy as np

from colabdesign.af.alphafold.common import residue_constants
import pandas as pd
import jax, pickle
import jax.numpy as jnp
from scipy.special import expit as sigmoid

import matplotlib.pyplot as plt
from scipy.special import softmax
import copy
from colabdesign.shared.protein import renum_pdb_str
from colabdesign.af.alphafold.common import protein

aa_order = {v:k for k,v in residue_constants.restype_order.items()}

def get_pdb(pdb_code=""):
  if pdb_code is None or pdb_code == "":
    upload_dict = files.upload()
    pdb_string = upload_dict[list(upload_dict.keys())[0]]
    with open("tmp.pdb","wb") as out: out.write(pdb_string)
    return "tmp.pdb"
  elif os.path.isfile(pdb_code):
    return pdb_code
  elif len(pdb_code) == 4:
    os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
    return f"{pdb_code}.pdb"
  else:
    os.system(f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v4.pdb")
    return f"AF-{pdb_code}-F1-model_v4.pdb"

def af2bind(outputs, mask_sidechains=True, seed=0):
  pair_A = outputs["representations"]["pair"][:-20,-20:]
  pair_B = outputs["representations"]["pair"][-20:,:-20].swapaxes(0,1)
  pair_A = pair_A.reshape(pair_A.shape[0],-1)
  pair_B = pair_B.reshape(pair_B.shape[0],-1)
  x = np.concatenate([pair_A,pair_B],-1)

  # get params
  if mask_sidechains:
    model_type = f"split_nosc_pair_A_split_nosc_pair_B_{seed}"
  else:
    model_type = f"split_pair_A_split_pair_B_{seed}"
  with open(f"af2bind_params/attempt_7_2k_lam0-03/{model_type}.pickle","rb") as handle:
    params_ = pickle.load(handle)
  params_ = dict(**params_["~"], **params_["linear"])
  p = jax.tree_map(lambda x:np.asarray(x), params_)

  # get predictions
  x = (x - p["mean"]) / p["std"]
  x = (x * p["w"][:,0]) + (p["b"] / x.shape[-1])
  p_bind_aa = x.reshape(x.shape[0],2,20,-1).sum((1,3))
  p_bind = sigmoid(p_bind_aa.sum(-1))
  return {"p_bind":p_bind, "p_bind_aa":p_bind_aa}
