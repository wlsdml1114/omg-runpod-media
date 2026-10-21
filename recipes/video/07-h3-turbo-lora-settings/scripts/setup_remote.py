"""Run inside the Pod. Installs pinned ComfyUI, H3 FL2VA weights and Turbo LoRAs, copies the x0capture node.
Copy this recipe folder to /workspace/h3-turbo-ablation first and put your first-frame image at inputs/real-hd-01-face.png."""
import concurrent.futures,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path('/workspace/h3-turbo-ablation');COMFY=ROOT/'ComfyUI'
def run(a,cwd=None):
    print('RUN',' '.join(map(str,a)),flush=True);subprocess.run(list(map(str,a)),cwd=cwd,check=True)
(ROOT/'logs').mkdir(parents=True,exist_ok=True)
if sys.prefix==sys.base_prefix:
    env=ROOT/'env'
    if not env.exists():run([sys.executable,'-m','venv','--system-site-packages',env])
    os.execv(str(env/'bin/python'),[str(env/'bin/python'),str(Path(__file__).resolve())])
if not COMFY.exists():
    run(['git','init',COMFY]);run(['git','-C',COMFY,'remote','add','origin','https://github.com/Comfy-Org/ComfyUI.git'])
sha=json.loads((ROOT/'inputs/source-revisions.json').read_text())['ComfyUI']
run(['git','-C',COMFY,'fetch','--depth','1','origin',sha]);run(['git','-C',COMFY,'checkout','--detach','FETCH_HEAD'])
import torch
pins=['torch=='+torch.__version__]
for p in ['torchvision','torchaudio']:
    m=__import__(p);pins.append(p+'=='+m.__version__)
c=ROOT/'logs/runtime-constraints.txt';c.write_text('\n'.join(pins)+'\n')
run([sys.executable,'-m','pip','install','--no-cache-dir','-r',COMFY/'requirements.txt','-c',c,'huggingface_hub[hf_xet]','aiohttp'])
for n in ['input','output','user']:(COMFY/n).mkdir(exist_ok=True)
shutil.copy2(ROOT/'inputs/real-hd-01-face.png',COMFY/'input/real-hd-01-face.png')
dst=COMFY/'custom_nodes/x0capture'
if dst.exists():shutil.rmtree(dst)
shutil.copytree(ROOT/'custom_nodes/x0capture',dst)
from huggingface_hub import hf_hub_download
man=json.loads((ROOT/'inputs/model-download-manifest.json').read_text())
def dl(it):
    if it['dest']=='loras':
        p=hf_hub_download(repo_id=it['repo'],filename=it['rfilename'],revision=it['revision'],local_dir=ROOT/'hf-lightx2v')
        t=COMFY/'models/loras'/it['rfilename'];t.parent.mkdir(parents=True,exist_ok=True)
        if not t.exists():os.symlink(p,t)
    else:
        p=hf_hub_download(repo_id=it['repo'],filename=it['rfilename'],revision=it['revision'],local_dir=COMFY/'models')
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==it['sha256'],it['rfilename']
    print('VERIFIED',it['rfilename'],flush=True)
    return {'file':it['rfilename'],'bytes':Path(p).stat().st_size,'sha256':h.hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:hashes=list(pool.map(dl,man['files']))
(ROOT/'logs/model-hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
v={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'vram_bytes':torch.cuda.get_device_properties(0).total_memory,'comfy_commit':sha}
(ROOT/'logs/runtime-versions.json').write_text(json.dumps(v,indent=2)+'\n')
with (ROOT/'logs/pip-freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
print('SETUP_COMPLETE',json.dumps(v),flush=True)
