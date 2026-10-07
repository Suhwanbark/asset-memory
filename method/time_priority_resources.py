"""Guard this task's subprocesses; never stop another user's process."""
from pathlib import Path
import argparse, json, os, shutil, signal, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
FLOOR = 20 * 1024**3  # stricter than decimal 20 GB

def disk_check(extra=0):
    free = shutil.disk_usage(ROOT).free
    if free < FLOOR + extra:
        raise RuntimeError(f"STOP: free={free} bytes; minimum={FLOOR+extra} bytes")
    return free

def gpu_check(uuid):
    out = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    row = next(x for x in out.splitlines() if x.split(',')[0].strip()==uuid)
    _, memory, util = [x.strip() for x in row.split(',')]
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'], text=True)
    if int(memory)>100 or int(util)>0 or any(x.split(',')[0].strip()==uuid for x in apps.splitlines()):
        raise RuntimeError(f"STOP: selected GPU is no longer idle: {row}")
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument('--gpu'); p.add_argument('command',nargs=argparse.REMAINDER)
    a=p.parse_args(); command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command: p.error('command required')
    start_free=disk_check(); env=os.environ.copy()
    if a.gpu:
        gpu_check(a.gpu); env['CUDA_VISIBLE_DEVICES']=a.gpu
    else: env['CUDA_VISIBLE_DEVICES']=''
    log=ROOT/'result/time_priority_v1/resource_events.jsonl'; log.parent.mkdir(parents=True,exist_ok=True)
    child=subprocess.Popen(command,env=env,start_new_session=True)
    try:
        with log.open('a') as f:
            f.write(json.dumps({'event':'start','time':time.time(),'pid':child.pid,'gpu':a.gpu,'free':start_free,'command':command})+'\n'); f.flush()
            while child.poll() is None:
                try: free=disk_check()
                except RuntimeError:
                    f.write(json.dumps({'event':'disk_floor_stop','time':time.time(),'pid':child.pid})+'\n'); f.flush()
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL)
                    raise
                time.sleep(2)
            f.write(json.dumps({'event':'exit','time':time.time(),'pid':child.pid,'code':child.returncode,'free':disk_check()})+'\n')
        sys.exit(child.returncode)
    except BaseException:
        if child.poll() is None: os.killpg(child.pid,signal.SIGTERM)
        raise

if __name__=='__main__': main()
