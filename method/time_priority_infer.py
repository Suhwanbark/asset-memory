"""Direct local A/B inference. Gold is never read by this module."""
from pathlib import Path
import argparse, hashlib, json, os, re, time
from time_priority_resources import ROOT, disk_check

DATA=ROOT/'data/time_priority_v1'; OUT=ROOT/'result/time_priority_v1'
SYSTEM=('You are an evidence analyst. Choose which of two dated information items deserves greater weight for the stated decision at the stated time. '
        'Use the supplied facts and explicit scenario assumptions. Both items can be true; this is not a truth-versus-falsehood test. '
        'Return A or B followed by a brief one-sentence justification. Do not provide a long chain of reasoning.')
SYSTEM_V2=('You are an evidence analyst. Choose which of two dated information items deserves greater weight for the stated decision at the stated time. '
           'Use the supplied facts and explicit scenario assumptions. Both items can be true; this is not a truth-versus-falsehood test. '
           'Give one short sentence explaining the decisive evidence. Then write your final choice on a new line in exactly this format: Answer: A or Answer: B. '
           'Do not provide a long chain of reasoning.')
SYSTEM_V3='Select the option that best answers the question. Respond with the option letter only.'

def parse_answer(text, version):
    if version in ['v1','v3']:
        m=re.match(r'^(?:Answer\s*:\s*)?\(?([AB])(?:\b|[.:-])',text,re.I)
        return m.group(1).upper() if m else None
    matches=re.findall(r'^\s*(?:\*\*)?Answer\s*:\s*([AB])\b',text,re.I|re.M)
    return matches[-1].upper() if matches else None

def parse_terminal_answer(text):
    """Whitespace-only format normalization; never infer a label from rationale."""
    match=re.search(r'(?:\*\*)?\bAnswer\s*:\s*(?:\*\*)?([AB])\b(?:\*\*)?[.!]?\s*$',text,re.I)
    return match.group(1).upper() if match else None

def sha(t): return hashlib.sha256(t.encode() if isinstance(t,str) else t).hexdigest()
def user_prompt(case,reverse=False,version='v1'):
    items=case['items'][::-1] if reverse else case['items']
    lines=[f"Assessment date: {case['as_of']}",f"Decision: {case['question']}",'']
    for label,item in zip(['A','B'],items):
        lines += [f"{label}. Published {item['published']}: {item['text']}",'']
    ending = {'v1': 'Which item deserves priority for this decision? Answer A or B, then briefly explain.',
              'v2': 'Which item deserves priority for this decision? Give one short sentence, then a final line: Answer: A or Answer: B.',
              'v3': 'Which item deserves priority for this decision? Reply with the option letter only.'}
    lines += [ending[version]]
    return '\n'.join(lines),[x['id'] for x in items]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-index',type=int,required=True); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--limit',type=int); p.add_argument('--version',choices=['v1','v2','v3'],default='v1')
    a=p.parse_args(); disk_check()
    assert os.environ.get('CUDA_VISIBLE_DEVICES','').startswith('GPU-'),'exactly one UUID required'
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES']
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    assert torch.cuda.device_count()==1
    torch.manual_seed(37)
    manifest=json.loads((DATA/'model_manifest.json').read_text())[a.model_index]
    frozen=json.loads((DATA/'freeze.json').read_text()); raw=(DATA/'cases.json').read_bytes()
    assert sha(raw)==frozen['cases_sha256']
    cases=json.loads(raw)['cases']; cases=cases[:a.limit] if a.limit else cases
    name=manifest['id'].split('/')[-1]; folder=(OUT if a.version=='v1' else OUT/a.version)/name; folder.mkdir(parents=True,exist_ok=True)
    print('LOAD',name,'GPU',os.environ['CUDA_VISIBLE_DEVICES'],flush=True)
    tok=AutoTokenizer.from_pretrained(manifest['path'],local_files_only=True,padding_side='left')
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(manifest['path'],local_files_only=True,torch_dtype=torch.float16,attn_implementation='sdpa').to('cuda:0').eval()
    param_count=sum(p.numel() for p in model.parameters()); assert param_count<=3_000_000_000,param_count
    jobs=[]
    for c in cases:
        for reverse in [False,True]:
            prompt,order=user_prompt(c,reverse,a.version)
            messages=[{'role':'system','content':{'v1':SYSTEM,'v2':SYSTEM_V2,'v3':SYSTEM_V3}[a.version]},{'role':'user','content':prompt}]
            full=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            path=folder/(c['id']+('__BA' if reverse else '__AB')+'.json')
            if path.exists():
                cached=json.loads(path.read_text()); assert cached['prompt_sha256']==sha(full); assert cached['revision']==manifest['revision']; continue
            jobs.append((c,reverse,prompt,order,full,path))
    ids={c:tok.encode(c,add_special_tokens=False) for c in ['A','B']}; assert all(len(x)==1 for x in ids.values()),ids
    print('READY',name,'params',param_count,'pending',len(jobs),flush=True)
    for start in range(0,len(jobs),a.batch_size):
        disk_check(); batch=jobs[start:start+a.batch_size]
        tensors=tok([x[4] for x in batch],padding=True,return_tensors='pt',truncation=False).to('cuda:0')
        assert tensors['input_ids'].shape[1]<=3000, 'do not silently truncate small-model context'
        t0=time.time()
        max_tokens={'v1':110,'v2':160,'v3':8}[a.version]
        with torch.inference_mode():
            out=model.generate(**tensors,do_sample=False,max_new_tokens=max_tokens,pad_token_id=tok.pad_token_id,return_dict_in_generate=True,output_scores=True)
        scores=out.scores[0][:,[ids['A'][0],ids['B'][0]]].float().softmax(-1).cpu().tolist()
        seqs=out.sequences[:,tensors['input_ids'].shape[1]:]
        for j,(case,reverse,prompt,order,full,path) in enumerate(batch):
            text=tok.decode(seqs[j],skip_special_tokens=True).strip()
            answer=parse_answer(text,a.version)
            chosen=order[['A','B'].index(answer)] if answer else None
            result=dict(case_id=case['id'],reverse=reverse,model=manifest['id'],revision=manifest['revision'],parameter_count=param_count,
                gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],thinking=False,decoding='greedy',max_new_tokens=max_tokens,prompt_version=a.version,
                prompt=full,prompt_sha256=sha(full),dataset_sha256=frozen['cases_sha256'],response=text,
                answer=answer,chosen_item=chosen,shown_order=order,first_token_AB_probabilities=scores[j],
                likelihood_choice=order[int(scores[j][1]>scores[j][0])] if a.version in ['v1','v3'] else None,
                input_tokens=int(tensors['attention_mask'][j].sum()),output_tokens=int((seqs[j]!=tok.pad_token_id).sum()),
                batch_elapsed_seconds=time.time()-t0,created_at=time.time())
            disk_check(); path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print('BATCH',name,start+len(batch),'/',len(jobs),'seconds',round(time.time()-t0,2),'free_GiB',round(disk_check()/1024**3,1),flush=True)
    print('COMPLETE',name,'GPU_peak_GiB',round(torch.cuda.max_memory_allocated()/1024**3,2),flush=True)

if __name__=='__main__': main()
