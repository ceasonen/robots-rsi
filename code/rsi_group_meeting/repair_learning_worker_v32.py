"""Frozen public-input spatial caching and declared actor inference, not a teacher."""
from pathlib import Path
import json,sys,contextlib,time,os,traceback,hashlib
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev'
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    with contextlib.redirect_stdout(sys.stderr):
        import numpy as np,torch
        from dmpel_native_backend_v6 import NativeDMPELSource90
        from dmpel_actor_runtime_v7 import ActorRuntime
        from check_dmpel_actor_runtime_v7 import digest_parameters
        p=json.loads((W/'registration.json').read_text('utf8'));native=NativeDMPELSource90('cuda',threads=2);runtime=ActorRuntime(native,77210)
        parent=p['parent'];assert sha(parent['file'])==parent['sha256'];runtime.load(torch.load(parent['file'],map_location='cuda',weights_only=True)['adapter'])
        frozen=digest_parameters(runtime.frozen);assert frozen==p['frozen_pretrained_parameters_sha256']
        native.policy.policy_head.low_eval_noise=False;current_actor='parent';parameter_digest=digest_parameters(list(runtime.policy.named_parameters()))
    print(json.dumps(dict(ready=True,pid=os.getpid(),metadata=dict(frozen_pretrained_sha256=frozen,initial_all_parameter_digest=parameter_digest,actor='parent',frozen_spatial_only_cache=True))),flush=True)
    for line in sys.stdin:
        try:
            q=json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                command=q['command']
                if command=='close':
                    assert frozen==digest_parameters(runtime.frozen)
                    print(json.dumps(dict(closed=True,frozen_sha256=frozen)),flush=True);return
                if command=='load_actor':
                    actor=q['actor'];assert actor in ['parent','uniform_bc','local_bc']
                    item=parent if actor=='parent' else json.loads((W/'trained_actors.json').read_text('utf8'))[actor]
                    assert sha(item['file'])==item['sha256'];payload=torch.load(item['file'],map_location='cuda',weights_only=True);runtime.load(payload['adapter']);current_actor=actor
                    native.policy.policy_head.low_eval_noise=False;answer=dict(loaded=actor,sha256=item['sha256'],all_parameter_digest=digest_parameters(list(runtime.policy.named_parameters())))
                elif command=='reset':native.reset(int(q['seed']));answer=dict(reset=True,history_length=0,actor=current_actor)
                elif command in ['encode','query']:
                    path=Path(q['observation_file']).resolve();assert path.is_relative_to(W)
                    started=time.perf_counter()
                    with np.load(path) as raw:
                        if command=='query':
                            action,logp,feature,_=runtime.query(raw,q['language']);answer=dict(action=action.tolist(),log_probability=logp,history_length=len(native.policy.latent_queue),actor=current_actor)
                        else:
                            language=q['language'];assert current_actor=='parent'
                            for key in ['agentview_image','robot0_eye_in_hand_image']:assert raw[key].shape==(128,128,3) and raw[key].dtype==np.uint8
                            assert raw['robot0_joint_pos'].shape==(7,) and raw['robot0_gripper_qpos'].shape==(2,)
                            if language not in native._tokens:
                                tokens=native.tokenizer(text=[language],add_special_tokens=True,max_length=native.cfg.data.max_word_len,padding='max_length',return_attention_mask=True,return_tensors='pt')
                                native._tokens[language]={key:tokens[key][0] for key in ['attention_mask','input_ids']}
                            data=native.preprocess([raw],native._tokens[language],native.cfg)
                            with torch.no_grad(),runtime.context():
                                data=native.policy.preprocess_input(data,train_mode=False);feature=native.policy.spatial_encode(data)[0,0].cpu().numpy()
                            assert feature.shape==(4,768) and feature.dtype==np.float32 and np.isfinite(feature).all()
                            output=Path(q['feature_file']).resolve();assert output.is_relative_to(W);np.save(output,feature)
                            answer=dict(feature_file=str(output),feature_sha256=sha(output),spatial_encodings=1,policy_action_queries=0)
                    answer['query_seconds']=time.perf_counter()-started
                else:raise AssertionError(command)
            print(json.dumps(answer),flush=True)
        except BaseException as ex:
            traceback.print_exc(file=sys.stderr);print(json.dumps(dict(error=repr(ex))),flush=True);return
if __name__=='__main__':main()
