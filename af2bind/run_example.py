from af2bind import *

target_pdb = "6w70" #@param {type:"string"}
target_chain = "A" #@param {type:"string"}
#@markdown - Please indicate target pdb (or uniprot ID to download from AlphaFoldDB) and chain.
#@markdown - Leave pdb blank for custom upload prompt.
mask_sidechains = True
mask_sequence = False

target_pdb = target_pdb.replace(" ","")
target_chain = target_chain.replace(" ","")
if target_chain == "":
  target_chain = "A"

pdb_filename = get_pdb(target_pdb)

clear_mem()
af_model = mk_afdesign_model(protocol="binder", debug=True)
af_model.prep_inputs(pdb_filename=pdb_filename,
                     chain=target_chain,
                     binder_len=20,
                     rm_target_sc=mask_sidechains,
                     rm_target_seq=mask_sequence)

# split
r_idx = af_model._inputs["residue_index"][-20] + (1 + np.arange(20)) * 50
af_model._inputs["residue_index"][-20:] = r_idx.flatten()

af_model.set_seq("ACDEFGHIKLMNPQRSTVWY")
af_model.predict(verbose=False)

o = af2bind(af_model.aux["debug"]["outputs"],
            mask_sidechains=mask_sidechains)
pred_bind = o["p_bind"].copy()
pred_bind_aa = o["p_bind_aa"].copy()

#######################################################
labels = ["chain","resi","resn","p(bind)"]
data = []
for i in range(af_model._target_len):
  c = af_model._pdb["idx"]["chain"][i]
  r = af_model._pdb["idx"]["residue"][i]
  a = aa_order.get(af_model._pdb["batch"]["aatype"][i],"X")
  p = pred_bind[i]
  data.append([c,r,a,p])

df = pd.DataFrame(data, columns=labels)
df.to_csv('results.csv')

#data_table.enable_dataframe_formatter()
df_sorted = df.sort_values("p(bind)",ascending=False, ignore_index=True).rename_axis('rank').reset_index()
#display(data_table.DataTable(df_sorted, min_width=100, num_rows_per_page=15, include_index=False))

top_n = 15
top_n_idx = pred_bind.argsort()[::-1][:15]
pymol_cmd="select ch"+str(target_chain)+","
for n,i in enumerate(top_n_idx):
  p = pred_bind[i]
  c = af_model._pdb["idx"]["chain"][i]
  r = af_model._pdb["idx"]["residue"][i]
  pymol_cmd += f" resi {r}"
  if n < top_n-1:
    pymol_cmd += " +"

print("\n🧪Pymol Selection Cmd:")
print(pymol_cmd)
