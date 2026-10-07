from pathlib import Path
import json
from huggingface_hub import HfApi, snapshot_download
from time_priority_resources import ROOT, disk_check

MODELS=['Qwen/Qwen3-1.7B','HuggingFaceTB/SmolLM2-1.7B-Instruct']
def main():
    disk_check(9*1024**3)
    target=ROOT/'data/time_priority_v1'; target.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in MODELS:
        disk_check(4*1024**3)
        info=HfApi().model_info(name)
        print('DOWNLOAD',name,'revision',info.sha,flush=True)
        path=snapshot_download(name,revision=info.sha,cache_dir=target/'models',
              allow_patterns=['*.safetensors','*.json','*.model','*.txt','*.jinja','README.md'],max_workers=2)
        rows.append({'id':name,'revision':info.sha,'path':path})
        print('READY',name,'free_GiB',round(disk_check()/1024**3,2),flush=True)
    (target/'model_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
if __name__=='__main__': main()
