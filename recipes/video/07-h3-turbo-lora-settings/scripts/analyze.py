"""Local metrics on native outputs: motion, sharpness, seed diversity, audio stats."""
import json,subprocess,itertools,numpy as np
from pathlib import Path
R=Path(__file__).resolve().parent.parent
W,H=336,192
def frames(run):
    raw=subprocess.run(['ffmpeg','-loglevel','error','-i',str(R/'runs'/run/'native.mp4'),'-vf',f'scale={W}:{H}','-f','rawvideo','-pix_fmt','gray','-'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.uint8).reshape(-1,H,W).astype(np.float32)
def audio(run):
    raw=subprocess.run(['ffmpeg','-loglevel','error','-i',str(R/'runs'/run/'native.mp4'),'-ac','1','-ar','16000','-f','f32le','-'],capture_output=True,check=True).stdout
    return np.frombuffer(raw,np.float32)
def lapvar(f):
    l=f[1:-1,1:-1]*4-f[:-2,1:-1]-f[2:,1:-1]-f[1:-1,:-2]-f[1:-1,2:]
    return float(l.var())
out={}
runs=sorted(p.name for p in (R/'runs').iterdir() if (p/'native.mp4').exists())
cache={r:frames(r) for r in runs}
for r in runs:
    f=cache[r];a=audio(r)
    face=f[:, 20:150, 110:230]
    spec=np.abs(np.fft.rfft(a[:len(a)//512*512].reshape(-1,512)*np.hanning(512),axis=1))+1e-9
    flat=float(np.mean(np.exp(np.mean(np.log(spec),1))/np.mean(spec,1)))
    hf=float(spec[:,128:].sum()/spec.sum())  # energy above 4 kHz
    out[r]={'frames':len(f),'motion_mean_abs_diff':float(np.abs(np.diff(f,axis=0)).mean()),
            'drift_from_first':float(np.abs(f-f[0]).mean()),'face_sharpness_lapvar':float(np.mean([lapvar(x) for x in face[::8]])),
            'audio_rms':float(np.sqrt((a**2).mean())),'audio_spectral_flatness':flat,'audio_hf_ratio_4k':hf}
def div(group):
    fs=[cache[r] for r in group];n=min(len(x) for x in fs)
    return float(np.mean([np.abs(a[:n]-b[:n]).mean() for a,b in itertools.combinations(fs,2)]))
seeds=['s42001','s42002','s42003','s42004']
base=[r for r in runs if 'base-20st' in r];turbo=[r for r in runs if r.startswith(('E1-v10_4-4st','E4-v10_4-4st'))]
out['_seed_diversity']={'base20_pairwise_mean_abs_diff':div(base),'turbo4_pairwise_mean_abs_diff':div(turbo),'base_runs':base,'turbo_runs':turbo}
(R/'evidence/metrics.json').write_text(json.dumps(out,indent=1))
for r in runs:print(f"{r:30s} motion {out[r]['motion_mean_abs_diff']:5.2f} drift {out[r]['drift_from_first']:5.2f} sharp {out[r]['face_sharpness_lapvar']:7.1f} rms {out[r]['audio_rms']:.3f} flat {out[r]['audio_spectral_flatness']:.3f} hf {out[r]['audio_hf_ratio_4k']:.3f}")
print(out['_seed_diversity'])
