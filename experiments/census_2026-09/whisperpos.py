import numpy as np, torch, sys, itertools
from transformers import AutoFeatureExtractor, WhisperModel
from pilot import load_speech, SR
torch.set_grad_enabled(False)
name=sys.argv[1]; layer=int(sys.argv[2])
sp=load_speech(70); rng=np.random.default_rng(1)
fe=AutoFeatureExtractor.from_pretrained(name); m=WhisperModel.from_pretrained(name,attn_implementation="eager",use_safetensors=False).encoder.eval()
sets=[]
for k in range(4):
    order=rng.permutation(len(sp)); x=np.concatenate([sp[i] for i in order])[:30*SR]
    x=np.roll(x, rng.integers(0,SR*3))  # also shift content
    h=m(fe(x,sampling_rate=SR,return_tensors="pt").input_features,output_hidden_states=True).hidden_states[layer][0]
    n=h.norm(dim=-1).numpy(); idx=set(np.where(n>10*np.median(n))[0].tolist()); sets.append(idx)
    print(k,len(idx),sorted(idx)[:25])
J=[len(a&b)/len(a|b) for a,b in itertools.combinations(sets,2)]
common=set.intersection(*sets)
print("pairwise Jaccard of outlier positions across different 30s speech inputs:",np.round(J,2),"| in all 4:",len(common),sorted(common))
# chance level: expected Jaccard for random sets of same size among 1500
s=np.mean([len(a) for a in sets]); print("chance Jaccard ~", round(s/(2*1500-s),3))
