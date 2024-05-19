import sys
import torch
import esm

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

WT = sys.argv[1]

gap = "GGS"
if len(sys.argv)>2:
    gap = sys.argv[2]

repeat = 1
if len(sys.argv)>3:
    repeat = int(sys.argv[3])

data = []
for i in range(len(WT)+1):
    seq = WT[:i]+ gap*repeat + WT[i:]
    data.append(("prot"+str(i), seq))
print(data)

batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    #token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1)
    logits = model(batch_tokens, repr_layers=[], return_contacts=False)["logits"]
    log_probs = torch.log_softmax(logits, dim=-1)

# 获取目标序列的token ids
target_tokens = batch_tokens[:, 1:]

# 计算每个位置的NLL
nll = -log_probs.gather(dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)

# 计算每条序列的总NLL
sequence_nll = nll.sum(dim=1)

# 计算每条序列的概率
sequence_prob = torch.exp(-sequence_nll)

# 输出每条序列的负对数似然和概率
for i, (label, seq) in enumerate(data):
    #print(f"Sequence: {label}")
    #print(f"Negative Log-Likelihood: {sequence_nll[i].item()}")
    #print(f"Probability: {sequence_prob[i].item()}\n")
    print(f"insert= %d, nll= %6.4f" % (i, sequence_nll[i].item()))
