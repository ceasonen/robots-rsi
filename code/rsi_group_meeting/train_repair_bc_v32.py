"""Two matched twenty-step BC operations; no simulator outcomes are collected."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time,os
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def main():
    for k in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[k]='1'
    import numpy as np,torch
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    from dmpel_native_backend_v6 import NativeDMPELSource90
    from dmpel_actor_runtime_v7 import ActorRuntime
    from dmpel_onpolicy_objective_v7 import gmm_joint_kl
    from check_dmpel_actor_runtime_v7 import digest_parameters
    p=read(W/'registration.json');manifest=read(W/'training_membership.json');cache=read(W/'cache_complete.json')
    assert not (W/'training_started.json').exists() and not list(W.glob('evaluation/results/*'))
    for f,h in p['implementation_sha256'].items():assert sha(ROOT/f)==h
    assert manifest['registration_sha256']==cache['registration_sha256']==sha(W/'registration.json')
    for f,h in manifest['files_sha256'].items():assert sha(ROOT/f)==h
    arrays={}
    def sample(row):
        f=ROOT/row['file'];field=row['feature_field'];assert field in ['spatial_features','frozen_spatial_features']
        if str(f) not in arrays:
            with np.load(f) as z:arrays[str(f)]=dict(spatial_features=z[field].copy(),actions=z['actions'].copy(),source_field=field)
        a=arrays[str(f)];i=row['action_index'];assert i>=9
        assert a['source_field']==field
        return a['spatial_features'][i-9:i+1],a['actions'][i]
    with exclusive('robotics_gpu'),exclusive('repair_learning_v32'),awake():
        save(W/'training_started.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),registration_sha256=sha(W/'registration.json'),membership_sha256=sha(W/'training_membership.json')))
        native=NativeDMPELSource90('cuda',threads=2);runtime=ActorRuntime(native,77210);initial=torch.load(p['parent']['file'],map_location='cuda',weights_only=True)['adapter'];runtime.load(initial)
        frozen=digest_parameters(runtime.frozen);assert frozen==p['frozen_pretrained_parameters_sha256'];native.policy.policy_head.low_eval_noise=False;actors={}
        for operation in ['uniform_bc','local_bc']:
            runtime.load(initial);torch.manual_seed(20265450);torch.cuda.reset_peak_memory_stats();optimizer=torch.optim.AdamW(runtime.parameters,lr=1e-4,weight_decay=0)
            initial_adapter={k:v.clone() for k,v in runtime.state().items()};metrics=[];started=time.perf_counter();rng=np.random.default_rng(20265450)
            for step in range(20):
                optimizer.zero_grad(set_to_none=True);loss_sum=kl_sum=0.;teacher_nll=source_nll=0.
                for poolname in [operation,'retention']:
                    pool=manifest['selected_targets'][poolname];indices=rng.integers(0,len(pool),size=16)
                    for j in indices:
                        features,action=sample(pool[int(j)]);assert features.dtype==np.float32 and features.shape==(10,4,768)
                        history=torch.from_numpy(features.copy()).unsqueeze(0).to('cuda');target=torch.as_tensor(action,device='cuda',dtype=torch.float32).unsqueeze(0)
                        with torch.no_grad(),runtime.reference():reference=runtime.distribution(history)
                        current=runtime.distribution(history);nll=-current.log_prob(target);kl=gmm_joint_kl(current,reference);loss=(nll+.001*kl).sum()/32
                        assert torch.isfinite(loss) and loss.requires_grad;loss.backward();loss_sum+=float(loss.detach().cpu());kl_sum+=float(kl.detach().sum().cpu())/32
                        if poolname=='retention':source_nll+=float(nll.detach().sum().cpu())/16
                        else:teacher_nll+=float(nll.detach().sum().cpu())/16
                assert all(v.grad is None for _,v in runtime.frozen)
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in runtime.parameters)
                norm=torch.nn.utils.clip_grad_norm_(runtime.parameters,1.);assert torch.isfinite(norm);optimizer.step()
                assert all(torch.isfinite(v).all() for v in runtime.parameters)
                row=dict(step=step+1,loss=loss_sum,joint_kl_upper_bound=kl_sum,teacher_nll=teacher_nll,retention_nll=source_nll,gradient_norm_before_clip=float(norm.cpu()),teacher_presentations=16,retention_presentations=16)
                metrics.append(row);save(W/'training_metrics'/f'{operation}_step{step+1:02d}.json',row)
            optimizer.zero_grad(set_to_none=True);state=runtime.state();assert any(not torch.equal(state[k],initial_adapter[k]) for k in state)
            assert digest_parameters(runtime.frozen)==frozen
            file=W/'actors'/f'{operation}_step20.pt';file.parent.mkdir(parents=True,exist_ok=True);assert not file.exists();torch.save(dict(adapter=state),file)
            actors[operation]=dict(file=str(file),sha256=sha(file),steps=20,teacher_presentations=320,retention_presentations=320,distinct_teacher_target_entries=len(manifest['selected_targets'][operation]),distinct_retention_target_entries=len(manifest['selected_targets']['retention']),elapsed_seconds=time.perf_counter()-started,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),frozen_parameters_sha256=frozen,all_parameter_digest=digest_parameters(list(runtime.policy.named_parameters())),metrics=metrics)
            save(W/'training_receipts'/f'{operation}.json',actors[operation]);print('Completed declared20-step BC operation',operation,flush=True)
        save(W/'trained_actors.json',actors)
if __name__=='__main__':main()
