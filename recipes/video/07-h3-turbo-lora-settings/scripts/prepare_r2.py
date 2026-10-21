"""Rerun matrix R: Turbo at its training shift (video 6 / audio 3) across samplers. Appends to run-matrix.json."""
import json,copy
from pathlib import Path
R=Path(__file__).resolve().parent.parent
M=json.loads((R/'inputs/run-matrix.json').read_text())
M=[m for m in M if m['experiment']!='R']
base=json.loads((R/'workflows/runs/E1-v10_4-4st-s42001.api.json').read_text())
LORA={'v10_4':'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors','v12_4':'minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors'}
cells=[('v10_4','res_multistep',6,1.0,False),('v10_4','euler',6,1.0,False),('v10_4','er_sde',6,1.0,False),('v10_4','lcm',6,1.0,True),
       ('v10_4','er_sde',6,0.75,False),('v10_4','lcm',12,1.0,False),('v12_4','lcm',6,1.0,False),('v12_4','er_sde',6,1.0,False)]
ids=[]
for lora,smp,shift,w,x0 in cells:
    rid=f"R-{lora}-4st-{smp}-sh{shift}"+(f"-w{int(w*100):03d}" if w!=1.0 else '')+"-s42001"
    g=copy.deepcopy(base)
    g['140:134']['inputs'].update(lora_name=LORA[lora],strength_model=w)
    g['140:123']['inputs']['sampler_name']=smp
    g['140:200']={'class_type':'MiniMaxH3SigmaShift','inputs':{'model':['140:135',0],'shift_video':float(shift),'shift_audio':3.0}}
    g['140:124']['inputs']['model']=['140:200',0];g['140:126']['inputs']['model']=['140:200',0]
    g['92']['inputs']['filename_prefix']='h3turbo/'+rid
    s=g['140:125']
    if x0:s['class_type']='SamplerCustomAdvancedX0Steps';s['inputs']['save_dir']='/workspace/h3-turbo-ablation/x0/'+rid
    else:
        s['class_type']='SamplerCustomAdvanced';s['inputs'].pop('save_dir',None)
    (R/'workflows/runs'/f'{rid}.api.json').write_text(json.dumps(g,ensure_ascii=False,indent=1)+'\n')
    M.append(dict(run_id=rid,experiment='R',lora=lora,lora_file=LORA[lora],steps=4,seed=42001,strength=w,x0=x0,sampler=smp,shift_video=shift,shift_audio=3));ids.append(rid)
(R/'inputs/run-matrix.json').write_text(json.dumps(M,indent=1)+'\n')
(R/'inputs/r2-ids.txt').write_text('\n'.join(ids)+'\n')
print('\n'.join(ids))
