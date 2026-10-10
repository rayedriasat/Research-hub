import numpy as np, torch, sys
from transformers import AutoFeatureExtractor, WhisperModel
from pilot import load_speech, SR
torch.set_grad_enabled(False)
name=sys.argv[1]; layer=int(sys.argv[2])
sp=load_speech(70); rng=np.random.default_rng(1)
fe=AutoFeatureExtractor.from_pretrained(name); m=WhisperModel.from_pretrained(name,attn_implementation="eager",use_safetensors=False).encoder.eval()
allp=[]; auc=[]
for k in range(4):
    order=rng.permutation(len(sp)); x=np.concatenate([sp[i] for i in order])[:30*SR]; x=np.roll(x, rng.integers(0,SR*3))
    feats=fe(x,sampling_rate=SR,return_tensors="pt").input_features
    h=m(feats,output_hidden_states=True).hidden_states[layer][0]; n=h.norm(dim=-1).numpy()
    e=np.array([10*np.log10(np.mean(x[i*320:(i+1)*320]**2)+1e-10) for i in range(1500)])
    idx=np.where(n>10*np.median(n))[0]
    pct=[(e<e[j]).mean() for j in idx]; allp+=pct
    # AUROC of -energy for outlier status
    from itertools import product
    pos=e[idx]; neg=np.delete(e,idx)
    auc.append(np.mean([(p<q)+0.5*(p==q) for p in pos for q in neg]))
    print(k, "outlier frame energy percentiles:", np.round(sorted(pct),2)[:12], "... frac below 20th pct:", round(np.mean(np.array(pct)<0.2),2))
print("mean energy pct of outliers:", round(np.mean(allp),3), " AUROC(low energy -> outlier):", np.round(auc,3))
