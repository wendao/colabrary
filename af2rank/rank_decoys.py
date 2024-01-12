from af2rank import *

#@markdown ### **settings**
recycles = 1 #@param ["0", "1", "2", "3", "4"] {type:"raw"}
iterations = 1

# decide what model to use
model_mode = "alphafold" #@param ["alphafold", "alphafold-multimer"]
model_num = 1 #@param ["1", "2", "3", "4", "5"] {type:"raw"}

if model_mode == "alphafold":
  model_name = f"model_{model_num}_ptm"
if model_mode == "alphafold-multimer":
  model_name = f"model_{model_num}_multimer_v3"

save_output_pdbs = False #@param {type:"boolean"}

#@markdown ### **advanced**
mask_sequence = True #@param {type:"boolean"}
mask_sidechains = True #@param {type:"boolean"}
mask_interchain = False #@param {type:"boolean"}

SETTINGS = {"rm_seq":mask_sequence,
            "rm_sc":mask_sidechains,
            "rm_ic":mask_interchain,
            "recycles":int(recycles),
            "iterations":int(iterations),
            "model_name":model_name}

NAME = "1mjc"
CHAIN = "A" # this can be multiple chains
NATIVE_PATH = f"{NAME}.pdb"
DECOY_DIR = f"{NAME}"

if save_output_pdbs:
  os.makedirs(f"{NAME}_output",ok_exists=True)

# setup model
clear_mem()
af = af2rank(NATIVE_PATH, CHAIN, model_name=SETTINGS["model_name"])

# score no structure
_ = af.predict(pdb=NATIVE_PATH, input_template=False, **SETTINGS)

SCORES = []

# score native structure
SCORES.append(af.predict(pdb=NATIVE_PATH, **SETTINGS, extras={"id":NATIVE_PATH}))

# score the decoy sctructures
for decoy_pdb in os.listdir(DECOY_DIR):
  input_pdb = os.path.join(DECOY_DIR, decoy_pdb)
  if save_output_pdbs:
    output_pdb = os.path.join(f"{NAME}_output",decoy_pdb)
  else:
    output_pdb = None
  SCORES.append(af.predict(pdb=input_pdb, output_pdb=output_pdb,
                           **SETTINGS, extras={"id":decoy_pdb}))

plot_me(SCORES, x="tm_i", y="composite", title=f"{NAME}: ranking INPUT decoys using composite score", fn=NAME+"_input")

plot_me(SCORES, x="tm_o", y="ptm", title=f"{NAME}: ranking OUTPUT decoys using predicted TMscore", fn=NAME+"_output")

plot_me(SCORES, x="tm_i", y="tm_o", diag=True, title=f"{NAME}: improvements over input structure", fn=NAME+"_compare")


