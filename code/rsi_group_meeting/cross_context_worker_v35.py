"""Inference only: frozen parent/U/L, original learned sigma and explicit ACK."""
from pathlib import Path
import contextlib,json,sys,time,os,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v35/cross_context_diagnostic'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np,torch
        from dmpel_native_backend_v6 import NativeDMPELSource90
        from dmpel_actor_runtime_v7 import ActorRuntime
        from check_dmpel_actor_runtime_v7 import digest_parameters
        p=json.loads((W/'registration.json').read_text('utf8'))
        native=NativeDMPELSource90('cuda',threads=2);runtime=ActorRuntime(native,77210)
        frozen=digest_parameters(runtime.frozen);assert frozen==p['frozen_pretrained_parameters_sha256']
        actor='parent';item=p['actors'][actor];assert sha(item['file'])==item['sha256']
        runtime.load(torch.load(item['file'],map_location='cuda',weights_only=True)['adapter'])
        runtime.policy.policy_head.low_eval_noise=False
        parameters=digest_parameters(list(runtime.policy.named_parameters()));assert parameters==item['all_parameter_digest']
    print(json.dumps(dict(ready=True,pid=os.getpid(),metadata=dict(native=native.metadata,frozen_pretrained_sha256=frozen,initial_all_parameter_digest=parameters,actor=actor,inference_only=True))),flush=True)
    for line in sys.stdin:
        try:
            q=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                command=q['command']
                if command=='close':
                    assert digest_parameters(runtime.frozen)==frozen
                    parameters=digest_parameters(list(runtime.policy.named_parameters()));assert parameters==p['actors'][actor]['all_parameter_digest']
                    answer=dict(closed=True,frozen_sha256=frozen,actor=actor,all_parameter_digest=parameters)
                elif command=='load_actor':
                    actor=q['actor'];assert actor in p['actors'];item=p['actors'][actor]
                    assert sha(item['file'])==item['sha256'];runtime.load(torch.load(item['file'],map_location='cuda',weights_only=True)['adapter'])
                    runtime.policy.policy_head.low_eval_noise=False
                    parameters=digest_parameters(list(runtime.policy.named_parameters()));assert parameters==item['all_parameter_digest']
                    answer=dict(loaded=actor,sha256=item['sha256'],all_parameter_digest=parameters)
                elif command=='reset':
                    native.reset(int(q['seed']));answer=dict(reset=True,history_length=0,actor=actor,policy_seed=int(q['seed']),physical_resets=0)
                elif command=='query':
                    file=Path(q['observation_file']).resolve();assert file.is_relative_to(W)
                    packet=Path(q['feature_packet_file']).resolve();assert packet.is_relative_to(W/'feature_packets')
                    assert not packet.exists() and not packet.with_suffix('.json').exists()
                    assert q['query_phase'] in ['parity','case'] and isinstance(q['query_step'],int) and q['query_step']>=0
                    mode=q.get('decode_mode','learned');assert mode=='learned'
                    runtime.policy.policy_head.low_eval_noise=False;assert not runtime.policy.policy_head.training
                    started=time.perf_counter()
                    with np.load(file) as archive:raw={key:archive[key] for key in p['public_sensors']}
                    action,logp,feature,distribution=runtime.query(raw,q['language'])
                    scales=distribution.component_distribution.base_dist.scale
                    low=float(scales.min().cpu());high=float(scales.max().cpu());torch.cuda.synchronize()
                    assert np.isfinite(action).all() and np.isfinite(feature).all() and feature.shape==(4,768) and feature.dtype==np.float32
                    query_seconds=time.perf_counter()-started;history_length=len(native.policy.latent_queue)
                    writing=time.perf_counter();packet.parent.mkdir(parents=True,exist_ok=True)
                    # Preserve the alreadycomputed pre-action spatialfeature; noextra encoder/head/sample.
                    with packet.open('xb') as target:
                        np.savez_compressed(target,spatial_feature=feature,proposal_action=action,
                                            proposal_log_probability=np.asarray(logp,dtype=np.float64),
                                            language=np.asarray(q['language']),actor=np.asarray(actor),
                                            history_length=np.asarray(history_length,dtype=np.int64),
                                            case_key=np.asarray(q['query_case']),action_index=np.asarray(q['query_step'],dtype=np.int64),
                                            step=np.asarray(q['query_step']+1,dtype=np.int64),query_phase=np.asarray(q['query_phase']),
                                            decode_mode=np.asarray(mode),density_kind=np.asarray('policy_proposal_not_servo_behavior'))
                    packet_sha=sha(packet);receipt=packet.with_suffix('.json')
                    info=dict(feature_packet_file=str(packet),feature_packet_sha256=packet_sha,feature_shape=[4,768],feature_dtype='float32',
                              case_key=q['query_case'],action_index=q['query_step'],query_phase=q['query_phase'],language=q['language'],actor=actor,
                              history_length=history_length,decode_mode=mode,proposal_action=action.tolist(),proposal_log_probability=logp,
                              density_kind='policy_proposal_not_servo_behavior',packet_write_and_hash_seconds=time.perf_counter()-writing,
                              additional_encoder_or_head_or_sampling_calls=0,query_seconds_before_packet_io=query_seconds)
                    with receipt.open('x',encoding='utf8') as target:json.dump(info,target,indent=2)
                    receipt_sha=sha(receipt);io_seconds=time.perf_counter()-writing
                    answer=dict(action=action.tolist(),log_probability=logp,actor=actor,history_length=history_length,decode_mode=mode,normal_scale_min=low,normal_scale_max=high,query_seconds=query_seconds,
                                feature_packet_file=str(packet),feature_packet_sha256=packet_sha,feature_receipt_file=str(receipt),feature_receipt_sha256=receipt_sha,feature_packet_io_seconds=io_seconds)
                else:raise AssertionError(command)
            print(json.dumps(answer),flush=True)
            if command=='close':return
        except BaseException as error:
            traceback.print_exc(file=sys.stderr);print(json.dumps(dict(error=repr(error))),file=sys.__stdout__,flush=True);return
if __name__=='__main__':main()
