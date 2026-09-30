"""Pilot: per-token hidden-state L2 norms across layers for pretrained audio encoders.
Inputs: real speech (LibriSpeech dummy) padded with leading/trailing silence, plus synthetic noise.
Reports max/median norm ratio per layer, outlier fraction (>10x median), and where outliers sit
(silence vs speech, first/last frames)."""
import sys, json, numpy as np, torch
from transformers import AutoFeatureExtractor, AutoModel, WhisperModel, ASTModel
torch.set_grad_enabled(False)
SR=16000
def load_speech(n=6):
    import io, soundfile as sf
    from datasets import load_dataset, Audio
    ds=load_dataset("hf-internal-testing/librispeech_asr_dummy","clean",split="validation").cast_column("audio",Audio(decode=False))
    out=[]
    for i in range(n):
        a=ds[i]["audio"]; b=a.get("bytes") or open(a["path"],"rb").read()
        x,sr=sf.read(io.BytesIO(b),dtype="float32"); assert sr==SR; out.append(x)
    return out
def make_inputs(speech):
    rng=np.random.default_rng(0); X=[]
    for s in speech:
        s=s[:SR*6]; sil=np.zeros(SR*2,np.float32)
        X.append(("speech+sil", np.concatenate([sil,s,sil]), 2*SR, 2*SR+len(s)))
    for s1,s2 in zip(speech[:3],speech[3:6]):
        s1=s1[:SR*4]; s2=s2[:SR*4]; sil=np.zeros(SR*2,np.float32)
        # "sil_mid": speech | 2 s silence | speech  -> silence region is [len(s1), len(s1)+2SR)
        X.append(("sil_mid", np.concatenate([s1,sil,s2]), len(s1), len(s1)+2*SR))
    for s in speech[:3]:
        # trim leading/trailing low energy so speech touches both boundaries
        e=np.convolve(s**2,np.ones(400)/400,'same'); idx=np.where(e>1e-4)[0]; s=s[idx[0]:idx[-1]][:SR*6]
        X.append(("speech_only", s, 0, len(s)))
    X.append(("white_noise", (0.05*rng.standard_normal(SR*6)).astype(np.float32), 0, SR*6))
    return X
def stats(hs, mask_speech, atts=None):
    out=[]
    for l,h in enumerate(hs):
        a=h[0].abs(); mact=float(a.max()/a.median())
        extra={}
        if atts is not None and l>0 and atts[l-1] is not None:
            A=atts[l-1][0].float()  # heads x T x T
            recv=A.mean(0).mean(0).numpy()*A.shape[-1]  # received attention / uniform
            j=int(recv.argmax()); extra=dict(sink=round(float(recv.max()),1),sink_pos=j,sink_in_sil=bool(not mask_speech[j]) if j<len(mask_speech) else None,
                 sink_top1pct_mass=round(float(np.sort(recv)[::-1][:max(1,len(recv)//100)].sum()/len(recv)),3))
        n=h[0].norm(dim=-1).numpy(); med=np.median(n)
        ratio=float(n.max()/med); out_idx=np.where(n>10*med)[0]; top=int(n.argmax())
        frac_sil = float((~mask_speech[out_idx]).mean()) if len(out_idx) else float('nan')
        out.append(dict(layer=l,ratio=round(ratio,2),n_out=int(len(out_idx)),frac_out=round(len(out_idx)/len(n),4),
                        argmax=top,T=len(n),top_in_sil=bool(not mask_speech[top]),out_frac_sil=frac_sil,massive=round(mact,1),**extra))
    return out
def frame_mask(T, a, b, nsamp, offset=0):
    # map token index -> sample time (uniform), True=speech
    t=(np.arange(T)+0.5)/T*nsamp
    return (t>=a)&(t<b)
def run_frame_model(name, X):
    fe=AutoFeatureExtractor.from_pretrained(name); m=AutoModel.from_pretrained(name,attn_implementation="eager").eval()
    res=[]
    for tag,x,a,b in X:
        inp=fe(x,sampling_rate=SR,return_tensors="pt")
        o=m(**inp,output_hidden_states=True,output_attentions=True)
        hs=o.hidden_states; T=hs[0].shape[1]
        res.append((tag,stats(hs,frame_mask(T,a,b,len(x)),o.attentions)))
    return res
def run_whisper(name, X):
    fe=AutoFeatureExtractor.from_pretrained(name); m=WhisperModel.from_pretrained(name,attn_implementation="eager",use_safetensors=False).encoder.eval()
    res=[]
    for tag,x,a,b in X:
        inp=fe(x,sampling_rate=SR,return_tensors="pt")  # padded to 30 s
        o=m(inp.input_features,output_hidden_states=True,output_attentions=True); hs=o.hidden_states; T=hs[0].shape[1]
        # 30 s window: speech region in samples, padding is after len(x); last hidden state is post-final-LN -> skip it
        res.append((tag,stats(hs[:-1],frame_mask(T,a,b,30*SR),o.attentions)))
        print(tag, "input frames with audio:", round(len(x)/(30*SR)*T), "of", T)
    return res
def run_ast(name, X):
    fe=AutoFeatureExtractor.from_pretrained(name); m=ASTModel.from_pretrained(name,attn_implementation="eager").eval()
    res=[]
    for tag,x,a,b in X:
        inp=fe(x,sampling_rate=SR,return_tensors="pt")
        o=m(**inp,output_hidden_states=True,output_attentions=True); hs=[h[:,2:] for h in o.hidden_states]  # drop cls+dist
        atts=[A[:,:,2:,2:] for A in o.attentions]
        T=hs[0].shape[1]; fd,td=12,101  # AST 1024 frames -> 101 time x 12 freq patches
        tpos=np.arange(T)%td; nsamp=10.24*SR
        t=(tpos+0.5)/td*nsamp; mask=(t>=a)&(t<b)
        r=stats(hs,mask,atts)
        for l,A in enumerate(o.attentions):
            r[l+1]["attn_to_cls_dist"]=round(float(A[0][:,2:,:2].sum(-1).mean()),3)
        # also special tokens
        for l,h in enumerate(o.hidden_states):
            nn=h[0].norm(dim=-1); r[l]["cls_norm_over_med"]=round(float(nn[0]/nn[2:].median()),2); r[l]["dist_norm_over_med"]=round(float(nn[1]/nn[2:].median()),2)
        res.append((tag,r))
    return res
if __name__=="__main__":
    speech=load_speech(); X=make_inputs(speech)
    which=sys.argv[1]; name=sys.argv[2]
    fn={"frame":run_frame_model,"whisper":run_whisper,"ast":run_ast}[which]
    res=fn(name,X)
    json.dump(res,open(name.replace('/','_')+".json","w"))
    # summary: per layer, mean ratio across speech inputs, max outlier fraction, top-in-silence rate
    L=len(res[0][1])
    print(f"== {name}  layers={L-1}  tokens={res[0][1][0]['T']}")
    for l in range(L):
        sp=[r[l] for t,r in res if t=="speech+sil"]; nz=[r[l] for t,r in res if t=="white_noise"][0]
        extra=""
        if "cls_norm_over_med" in sp[0]: extra=f" cls/med={np.mean([s['cls_norm_over_med'] for s in sp]):.2f} dist/med={np.mean([s['dist_norm_over_med'] for s in sp]):.2f}"
        if "sink" in sp[0]: extra+=f" sink={np.mean([s['sink'] for s in sp]):.1f}x sinkpos={[s['sink_pos'] for s in sp][:4]} sink_in_sil={np.mean([bool(s['sink_in_sil']) for s in sp]):.2f} top1%mass={np.mean([s['sink_top1pct_mass'] for s in sp]):.2f}"
        if "attn_to_cls_dist" in sp[0]: extra+=f" attn->cls/dist={np.mean([s['attn_to_cls_dist'] for s in sp]):.3f}"
        sm=[r[l] for t,r in res if t=="sil_mid"]; so=[r[l] for t,r in res if t=="speech_only"]
        if "sink" in sp[0]:
            # for sil_mid: mask True means "in [a,b)" which here is the SILENCE region
            extra+=f" | silmid: sinkpos={[s['sink_pos'] for s in sm]} in_mid_sil={np.mean([not s['sink_in_sil'] for s in sm]):.2f} sink={np.mean([s['sink'] for s in sm]):.1f}x | speechonly: sinkpos={[s['sink_pos'] for s in so]}/{so[0]['T']} sink={np.mean([s['sink'] for s in so]):.1f}x"
        print(f"L{l:02d} massive={np.mean([s['massive'] for s in sp]):7.1f} max/med={np.mean([s['ratio'] for s in sp]):7.2f}  out>10x={np.mean([s['frac_out'] for s in sp])*100:5.2f}%  argmax_in_sil={np.mean([s['top_in_sil'] for s in sp]):.2f}  argmax_pos={[s['argmax'] for s in sp][:4]}  noise_ratio={nz['ratio']:.2f}{extra}")
