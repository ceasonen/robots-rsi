"""Frozen local parent inference; separate output directory and no training."""
from pathlib import Path
import contextlib,json,sys,time,os,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'rsi_group_meeting/v30/language_schedule_dev'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np,torch
        from dmpel_native_backend_v6 import NativeDMPELSource90
        from dmpel_actor_runtime_v7 import ActorRuntime
        from check_dmpel_actor_runtime_v7 import digest_parameters
        p=json.loads((WORK/'registration.json').read_text('utf8'));parent=p['parent'];assert sha(parent['file'])==parent['sha256']
        native=NativeDMPELSource90('cuda',threads=2);runtime=ActorRuntime(native,77210)
        payload=torch.load(parent['file'],map_location='cuda',weights_only=True);runtime.load(payload['adapter'])
        assert all(torch.count_nonzero(x)==0 for k,x in payload['adapter'].items() if k.endswith('lora_b'))
        frozen=digest_parameters(runtime.frozen)
        assert frozen==p['frozen_pretrained_parameters_sha256'];parameters=digest_parameters(list(runtime.policy.named_parameters()))
    print(json.dumps(dict(ready=True,pid=os.getpid(),metadata=dict(native=native.metadata,parent_sha256=parent['sha256'],frozen_pretrained_sha256=frozen,all_parameter_digest=parameters,inference_only=True))),flush=True)
    for line in sys.stdin:
        try:
            q=json.loads(line)
            if q['command']=='close':
                assert digest_parameters(list(runtime.policy.named_parameters()))==parameters
                print(json.dumps(dict(closed=True,all_parameter_digest=parameters)),flush=True);return
            with contextlib.redirect_stdout(sys.stderr):
                if q['command']=='reset':
                    native.reset(int(q['seed']));answer=dict(reset=True,history_length=0)
                elif q['command']=='query':
                    file=Path(q['observation_file']).resolve();assert file.is_relative_to(WORK)
                    started=time.perf_counter()
                    with np.load(file) as raw:action,logp,feature,distribution=runtime.query(raw,q['language'])
                    torch.cuda.synchronize();elapsed=time.perf_counter()-started
                    assert np.isfinite(action).all() and np.isfinite(feature).all()
                    answer=dict(action=action.tolist(),log_probability=logp,query_seconds=elapsed,history_length=len(native.policy.latent_queue),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
                else:raise ValueError(q['command'])
            print(json.dumps(answer),flush=True)
        except BaseException:
            traceback.print_exc(file=sys.stderr);print(json.dumps(dict(error='Local language-schedule worker stopped; preserve attempt; no automatic retry')),flush=True);return
if __name__=='__main__':main()
