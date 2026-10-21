"""On the Pod: wait for setup, start local ComfyUI, run the matrix, then decode saved per-step x0 latents."""
import datetime,hashlib,json,shutil,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path('/workspace/h3-turbo-ablation');COMFY=ROOT/'ComfyUI';S='http://127.0.0.1:8188'
PY=ROOT/'env/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def fetch(p,data=None):
    r=urllib.request.Request(S+p,data=None if data is None else json.dumps(data).encode(),headers={'Content-Type':'application/json'} if data else {})
    with urllib.request.urlopen(r,timeout=60) as f:return json.load(f)
def log(*a):print(now(),*a,flush=True)
while 'SETUP_COMPLETE' not in (ROOT/'logs/setup.log').read_text():
    if 'Traceback' in (ROOT/'logs/setup.log').read_text():sys.exit('setup failed')
    time.sleep(10)
cmd=[str(PY),str(COMFY/'main.py'),'--listen','127.0.0.1','--port','8188','--disable-auto-launch','--preview-method','none','--use-pytorch-cross-attention']
def _up():
    try:fetch('/system_stats');return True
    except Exception:return False
if not _up():srv=subprocess.Popen(cmd,cwd=COMFY,stdout=open(ROOT/'logs/comfy-server.log','w'),stderr=subprocess.STDOUT,start_new_session=True)
for _ in range(300):
    try:fetch('/system_stats');break
    except Exception:time.sleep(2)
info=fetch('/object_info');assert 'SamplerCustomAdvancedX0Steps' in info and 'LoadX0Step' in info,'x0capture node not loaded'
(ROOT/'logs/comfy-system-stats.json').write_text(json.dumps(fetch('/system_stats'),indent=1))
log('SERVER_READY')
def run_graph(rid,graph,outdir,limit=2400):
    outdir.mkdir(parents=True,exist_ok=True);(outdir/'workflow.api.json').write_text(json.dumps(graph,ensure_ascii=False,indent=1))
    rec={'run_id':rid,'started_at':now()};t=time.monotonic()
    try:
        acc=fetch('/prompt',{'prompt':graph})
        if acc.get('node_errors'):raise RuntimeError(json.dumps(acc['node_errors']))
        pid=acc['prompt_id'];hist=None
        while time.monotonic()-t<limit:
            h=fetch('/history/'+pid)
            if pid in h and (h[pid].get('status',{}).get('completed') or h[pid].get('status',{}).get('status_str')=='error'):hist=h[pid];break
            time.sleep(1)
        if hist is None:raise TimeoutError('timeout')
        (outdir/'comfy-history.json').write_text(json.dumps(hist,indent=1))
        if hist['status'].get('status_str')!='success':raise RuntimeError('execution error')
        mp4=[COMFY/'output'/i.get('subfolder','')/i['filename'] for o in hist['outputs'].values() for v in o.values() if isinstance(v,list) for i in v if isinstance(i,dict) and str(i.get('filename','')).endswith('.mp4')]
        shutil.copy2(mp4[-1],outdir/'native.mp4')
        rec.update(status='success',seconds=time.monotonic()-t,sha256=hashlib.sha256((outdir/'native.mp4').read_bytes()).hexdigest())
    except Exception as e:
        rec.update(status='failed',error=str(e),seconds=time.monotonic()-t)
    rec['finished_at']=now()
    with (ROOT/'runs.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    log(rec['status'].upper(),rid,round(rec['seconds'],1),rec.get('error',''))
    return rec
only=set(sys.argv[1:])
M=json.loads((ROOT/'inputs/run-matrix.json').read_text())
for m in M:
    if only and m['run_id'] not in only:continue
    if (ROOT/'runs'/m['run_id']/'native.mp4').exists():continue
    g=json.loads((ROOT/'workflows/runs'/f"{m['run_id']}.api.json").read_text())
    r=run_graph(m['run_id'],g,ROOT/'runs'/m['run_id'])
    if r['status']!='success' and m['experiment']=='E1':log('E1 failure; continuing')
    if m['x0']:
        for p in sorted((ROOT/'x0'/m['run_id']).glob('x0_step*.pt')):
            did=f"{m['run_id']}-{p.stem}"
            if (ROOT/'x0-decoded'/did/'native.mp4').exists():continue
            dg={'1':{'class_type':'LoadX0Step','inputs':{'path':str(p)}},
                '2':{'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_video_vae_fp16.safetensors'}},
                '3':{'class_type':'VAELoader','inputs':{'vae_name':'minimax_h3_audio_vae_fp32.safetensors'}},
                '4':{'class_type':'VAEDecode','inputs':{'samples':['1',0],'vae':['2',0]}},
                '5':{'class_type':'VAEDecodeAudio','inputs':{'samples':['1',0],'vae':['3',0]}},
                '6':{'class_type':'CreateVideo','inputs':{'fps':24,'bit_depth':8,'images':['4',0],'audio':['5',0]}},
                '7':{'class_type':'SaveVideo','inputs':{'filename_prefix':'h3turbo-x0/'+did,'format':'auto','codec':'auto','video':['6',0]}}}
            run_graph(did,dg,ROOT/'x0-decoded'/did,limit=900)
log('MATRIX_COMPLETE')
