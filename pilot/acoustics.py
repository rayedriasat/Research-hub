"""For each model json: at chosen layers, compute acoustic properties of the argmax-norm token and sink token:
frame energy percentile within the clip, neighbor cosine similarity of raw log-mel frames (redundancy)."""
import json, sys, numpy as np, librosa
sys.argv=[sys.argv[0]]+sys.argv[1:]
from pilot import load_speech, make_inputs, SR
X=make_inputs(load_speech())
def frame_feats(x, T, total):
    # map T tokens uniformly onto 'total' samples (whisper: 30 s window, others: len(x))
    xx=np.zeros(total,np.float32); xx[:len(x)]=x[:total]
    edges=np.linspace(0,total,T+1).astype(int)
    e=np.array([10*np.log10(np.mean(xx[a:b]**2)+1e-10) for a,b in zip(edges[:-1],edges[1:])])
    return e
for fn,total_fn,layers in [(f,None,None) for f in sys.argv[1:]]:
    r=json.load(open(fn)); L=len(r[0][1])
    isw='whisper' in fn
    print("==",fn)
    for li in sorted(set([L//2, (3*L)//4, L-1])):
        pct_arg=[];pct_sink=[];pos=[]
        for (tag,x,a,b),(tag2,st) in zip(X,r):
            s=st[li]; T=s['T']; total=30*SR if isw else len(x)
            if isw: T=int(round(len(x)/(30*SR)*T))  # restrict to audio frames
            e=frame_feats(x,T if not isw else s['T'],total)
            def pct(j): return float((e<e[j]).mean()) if j<len(e) else float('nan')
            pct_arg.append(pct(s['argmax']))
            if 'sink_pos' in s: pct_sink.append(pct(s['sink_pos']))
            pos.append((tag[:6],s['argmax'],s.get('sink_pos')))
        print(f" L{li}: argmax-token energy pct mean={np.nanmean(pct_arg):.2f}  sink-token energy pct mean={np.nanmean(pct_sink) if pct_sink else float('nan'):.2f}  {pos[:5]}")
