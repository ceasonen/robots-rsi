"""Inference sidecar for parent and two fixed standard distillation BC actors."""
from pathlib import Path
import contextlib,json,sys,time,os,traceback,hashlib
ROOT=Path('F:/robot').resolve()
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'rsi_group_meeting'))

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/v40/code/worker_state_coverage_v40.py','Preparation only; root must review/copy.'
    from state_coverage_study_v40 import W
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np,torch
        from dmpel_native_backend_v6 import NativeDMPELSource90
        from state_coverage_runtime_v40 import HeadOnlyRuntime
        from check_dmpel_actor_runtime_v7 import digest_parameters
        from paired_observation_v4 import array_hash
        p=json.loads((W/'registration.json').read_text('utf8'))
        actors=json.loads((W/'trained_actors.json').read_text('utf8'));assert set(actors)=={'oldA_KL','coverage_KL'}
        native=NativeDMPELSource90('cuda',threads=2);runtime=HeadOnlyRuntime(native,77210)
        frozen=digest_parameters(runtime.frozen);assert frozen==p['frozen_pretrained_parameters_sha256']
        parent_state=runtime.state();parent_digest=digest_parameters(list(runtime.policy.named_parameters()))
        assert all(v['initial_all_parameter_digest']==parent_digest for v in actors.values())
        assert torch.count_nonzero(runtime.layer.lora_b)==0 and runtime.layer.scale==2.
        expected={'parent':dict(p['parent'],all_parameter_digest=parent_digest),**actors}
        for item in expected.values():assert sha(item['file'])==item['sha256']
        actor='parent'
        class PolicyNumericalFailure(Exception):pass
        diagnostic={}
        def finite_linear(name):
            def hook(module,args,out):
                if not torch.isfinite(out).all():
                    diagnostic['layer_name']=name;diagnostic['nonfinite_head_output']=out.detach().float().cpu().numpy()
                    raise PolicyNumericalFailure('nonfinite policy-head linear output: '+name)
            return hook
        # These hooks inspect existing values, without extra forward or random calls.
        handles=[m.register_forward_hook(finite_linear(n)) for n,m in runtime.policy.policy_head.named_modules() if isinstance(m,torch.nn.Linear)]
    print(json.dumps(dict(ready=True,pid=os.getpid(),metadata=dict(native=native.metadata,frozen_pretrained_sha256=frozen,parent_all_parameter_digest=parent_digest,initial_all_parameter_digest=parent_digest,actor='parent',actor_parameter_digests={k:v['all_parameter_digest'] for k,v in expected.items()},inference_only=True,numerical_failure_hooks='inspect_existing_head_linear_outputs_no_extra_forward'))),flush=True)
    for line in sys.stdin:
        try:
            q=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                command=q['command']
                if command=='close':
                    assert digest_parameters(runtime.frozen)==frozen
                    actual=digest_parameters(list(runtime.policy.named_parameters()));assert actual==expected[actor]['all_parameter_digest']
                    answer=dict(closed=True,frozen_sha256=frozen,actor=actor,all_parameter_digest=actual,actor_parameter_digests={k:v['all_parameter_digest'] for k,v in expected.items()})
                elif command=='load_actor':
                    actor=q['actor'];assert actor in expected;item=expected[actor];assert sha(item['file'])==item['sha256']
                    if actor=='parent':runtime.load(parent_state)
                    else:runtime.load(torch.load(item['file'],map_location='cuda',weights_only=True)['adapter'])
                    runtime.layer.scale=2.;runtime.policy.policy_head.low_eval_noise=False
                    actual=digest_parameters(list(runtime.policy.named_parameters()));assert actual==item['all_parameter_digest']
                    answer=dict(loaded=actor,actor=actor,sha256=item['sha256'],all_parameter_digest=actual,logical_parent_file_not_loaded_into_head_layout=actor=='parent')
                elif command=='reset':
                    native.reset(int(q['seed']));answer=dict(reset=True,history_length=0,actor=actor,policy_seed=int(q['seed']),physical_resets=0)
                elif command=='query':
                    file=Path(q['observation_file']).resolve();assert file.is_relative_to(W)
                    packet=Path(q['feature_packet_file']).resolve();assert packet.is_relative_to(W/'feature_packets')
                    assert not packet.exists() and not packet.with_suffix('.json').exists()
                    assert q['query_phase']=='case' and q['query_step']>=0
                    runtime.policy.policy_head.low_eval_noise=False;assert not runtime.policy.policy_head.training
                    with np.load(file,allow_pickle=False) as z:raw={key:z[key].copy() for key in p['public_sensors']}
                    diagnostic.clear();started=time.perf_counter();failure=None;values={};sample_completed=False
                    try:
                        action,logp,feature,distribution=runtime.query(raw,q['language']);sample_completed=True
                        values=dict(spatial_feature=feature,proposal_action=action,proposal_log_probability=np.asarray(logp,dtype=np.float64))
                        if not(np.isfinite(action).all() and np.isfinite(feature).all() and np.isfinite(logp)):
                            raise PolicyNumericalFailure('nonfinite returned action, feature or log probability')
                        assert action.shape==(7,) and feature.shape==(4,768) and feature.dtype==np.float32
                        scales=distribution.component_distribution.base_dist.scale
                        scale_min=float(scales.min().cpu());scale_max=float(scales.max().cpu())
                        if not(np.isfinite(scale_min) and np.isfinite(scale_max)):
                            raise PolicyNumericalFailure('nonfinite returned proposal scale')
                    except PolicyNumericalFailure as err:
                        failure=str(err);values.update(diagnostic)
                        # Retain exact public inputs for this failed query; never fabricate an action.
                        values.update({'failure_public_'+k:v for k,v in raw.items()})
                    torch.cuda.synchronize();query_seconds=time.perf_counter()-started
                    history_length=len(native.policy.latent_queue);writing=time.perf_counter();packet.parent.mkdir(parents=True,exist_ok=True)
                    common=dict(language=np.asarray(q['language']),actor=np.asarray(actor),history_length=np.asarray(history_length,dtype=np.int64),case_key=np.asarray(q['query_case']),action_index=np.asarray(q['query_step'],dtype=np.int64),step=np.asarray(q['query_step']+1,dtype=np.int64),query_phase=np.asarray('case'),decode_mode=np.asarray('learned'),density_kind=np.asarray('policy_proposal'),policy_numerical_failure=np.asarray(failure is not None))
                    with packet.open('xb') as target:np.savez_compressed(target,**common,**values)
                    packet_sha=sha(packet);receipt=packet.with_suffix('.json')
                    info=dict(feature_packet_file=str(packet),feature_packet_sha256=packet_sha,case_key=q['query_case'],action_index=q['query_step'],query_phase='case',language=q['language'],actor=actor,history_length=history_length,decode_mode='learned',density_kind='policy_proposal',policy_numerical_failure=failure is not None,numerical_failure_reason=failure,action_sample_completed=sample_completed,public_pre_action_sensor_sha256=array_hash(raw),additional_encoder_or_head_or_sampling_calls=0,query_seconds_before_packet_io=query_seconds,packet_write_and_hash_seconds=time.perf_counter()-writing)
                    if failure is None:info.update(feature_shape=[4,768],feature_dtype='float32',proposal_action=action.tolist(),proposal_log_probability=float(logp))
                    with receipt.open('x',encoding='utf8') as target:json.dump(info,target,indent=2,allow_nan=False)
                    answer=dict(actor=actor,history_length=history_length,decode_mode='learned',query_seconds=query_seconds,policy_numerical_failure=failure is not None,numerical_failure_reason=failure,action_sample_completed=sample_completed,public_pre_action_sensor_sha256=info['public_pre_action_sensor_sha256'],feature_packet_file=str(packet),feature_packet_sha256=packet_sha,feature_receipt_file=str(receipt),feature_receipt_sha256=sha(receipt),feature_packet_io_seconds=time.perf_counter()-writing)
                    if failure is None:answer.update(action=action.tolist(),log_probability=float(logp),normal_scale_min=scale_min,normal_scale_max=scale_max)
                else:raise AssertionError(command)
            print(json.dumps(answer,allow_nan=False),flush=True)
            if command=='close':return
        except BaseException as error:
            traceback.print_exc(file=sys.stderr)
            print(json.dumps(dict(error=repr(error),unclassified_failure_requires_infrastructure_audit=True)),file=sys.__stdout__,flush=True)
            return
if __name__=='__main__':main()
