import torch
ckpt = torch.load("runs/checkpoints/core_gru.pt", map_location="cpu")
print(ckpt.keys())          # e.g., dict_keys(['gru','halt_head','update_head','config'])
print(ckpt['config'])       # saved config dict
# To see tensor shapes:
for k,v in ckpt.items():
    if hasattr(v, 'items'):
        continue
    print(k, {subk: subv.shape for subk, subv in v.items()})
