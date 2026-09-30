import numpy as np, torch, sys
from transformers import AutoFeatureExtractor, WhisperModel
from pilot import load_speech, SR
torch.set_grad_enabled(False)
name=sys.argv[1]
sp=load_speech(40)
full=np.concatenate(sp)[:30*SR]
assert len(full)==30*SR, len(full)
gap=full.copy(); gap[12*SR:17*SR]=0      # 5 s digital silence in the middle
noisegap=full.copy(); noisegap[12*SR:17*SR]=0.003*np.random.default_rng(0).standard_normal(5*SR)
short=full[:5*SR]
fe=AutoFeatureExtractor.from_pretrained(name); m=WhisperModel.from_pretrained(name,attn_implementation="eager",use_safetensors=False).encoder.eval()
for tag,x in [("full30_speech",full),("full30_gap12-17s",gap),("full30_noisegap12-17s",noisegap),("5s_speech+25s_pad",short)]:
    inp=fe(x,sampling_rate=SR,return_tensors="pt")
    o=m(inp.input_features,output_hidden_states=True,output_attentions=True)
    print("##",tag)
    for l,h in enumerate(o.hidden_states[:-1]):
        n=h[0].norm(dim=-1).numpy(); med=np.median(n); idx=np.where(n>10*med)[0]
        A=o.attentions[l-1][0].mean(0).mean(0).numpy()*n.shape[0] if l>0 else None
        s=f"  L{l} max/med={n.max()/med:6.1f} n_out={len(idx):3d} argmax={int(n.argmax())}"
        if len(idx): s+=f" out_frames(s)=[{idx.min()/50:.1f}..{idx.max()/50:.1f}] sample={list((idx[:8]/50).round(2))}"
        if A is not None: s+=f" sink={A.max():.1f}x@{int(A.argmax())/50:.2f}s"
        print(s)
