"""Build the run matrix and one ComfyUI API graph per run from the 2026-09-10 FL2VA template."""
import json,copy
from pathlib import Path
R=Path(__file__).resolve().parent.parent
T=json.loads((R/'workflows/template-turbo4-i2v.api.json').read_text())
PROMPT=("One continuous five-second live-action shot, fixed camera, no cuts. The adult Korean woman from the first frame stays seated in the same softly lit room, "
"looks into the camera and says in Korean, ‘오늘은 터보 로라를 켜고 꺼 볼게요.’ with natural lip sync. "
"While speaking she tilts her head slightly to her left, blinks once and finishes with a small closed-mouth smile. Keep her face, hair strands, cardigan knit, window light and room geometry stable. "
"Natural Korean female voice, close and clear, quiet indoor room tone, no music, no subtitles, no on-screen text.")
LORAS={'v10_4':'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors','v11_4':'minimax_h3_fl2v_turbo_4step_v1.1_768p_comfyui_bf16.safetensors',
'v12_4':'minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors','v10_8':'minimax_h3_fl2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors'}
M=[]
def add(exp,lora,steps,seed=42001,strength=1.0,x0=False):
    rid=f"{exp}-{'base' if lora is None else lora}-{steps}st"+(f"-w{int(strength*100):03d}" if lora and strength!=1.0 else '')+f"-s{seed}"
    M.append(dict(run_id=rid,experiment=exp,lora=lora,lora_file=LORAS.get(lora),steps=steps,seed=seed,strength=strength,x0=x0))
# order: fast turbo cells first, base 20-step last
add('E1',None,4,x0=True); add('E1','v10_4',4,x0=True)
add('E2','v10_4',2); add('E2','v10_4',8); add('E2','v10_8',4); add('E2','v10_8',8)
for w in (0.25,0.5,0.75): add('E3','v10_4',4,strength=w)
add('E6','v11_4',4); add('E6','v12_4',4)
for s in (42002,42003,42004): add('E4','v10_4',4,seed=s)
add('E1',None,20,x0=True)
for s in (42002,42003,42004): add('E4',None,20,seed=s)
(R/'workflows/runs').mkdir(parents=True,exist_ok=True)
for m in M:
    w=copy.deepcopy(T)
    w.pop('115')
    n=w['140:131']['inputs']; n.update(prompt=PROMPT,width=1344,height=768)
    w['140:127']['inputs']['unet_name']='minimax_h3_fl2va_pruned_int8_convrot.safetensors'
    w['140:129']['inputs']['noise_seed']=m['seed']
    w['140:139']['inputs']['value']=m['lora'] is not None
    w['140:138']['inputs']['value']=m['steps']; w['140:137']['inputs']['value']=m['steps']
    w['140:134']['inputs'].update(lora_name=m['lora_file'] or LORAS['v10_4'],strength_model=m['strength'])
    w['141']['inputs']['image']='real-hd-01-face.png'
    w['92']['inputs']['filename_prefix']='h3turbo/'+m['run_id']
    if m['x0']:
        s=w['140:125']; s['class_type']='SamplerCustomAdvancedX0Steps'; s['inputs']['save_dir']='/workspace/h3-turbo-ablation/x0/'+m['run_id']
    (R/'workflows/runs'/f"{m['run_id']}.api.json").write_text(json.dumps(w,ensure_ascii=False,indent=1)+'\n')
(R/'inputs/run-matrix.json').write_text(json.dumps(M,indent=1)+'\n')
(R/'prompts.json').write_text(json.dumps({'prompt':PROMPT,'first_frame':'inputs/real-hd-01-face.png','width':1344,'height':768,'frames':124,'fps':24,'sampler':'res_multistep','scheduler':'simple','guider':'BasicGuider (no CFG; released H3 weights are CFG-distilled)'},ensure_ascii=False,indent=1)+'\n')
print(len(M),'runs;',sum(m['steps'] for m in M),'total steps')
for m in M: print(m['run_id'])
