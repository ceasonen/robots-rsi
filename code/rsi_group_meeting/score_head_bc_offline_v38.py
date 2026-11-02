"""Exactly3291 new-head density forwards; all13164 old actor scores are reused."""
import sys,os,time,math,traceback,json
from pathlib import Path
from head_bc_study_v38 import ROOT,W,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'scripts'))
def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/score_head_bc_offline_v38.py','Preparation only; root must review/copy before execution.'
    p=verify();analysis=read(W/'analysis/head_readout.json');assert len(analysis['rows'])==12
    assert read(W/'evaluation_complete.json')['physical_episodes']==12 and not (W/'offline_fit').exists()
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    folder=W/'offline_fit'
    with exclusive('robotics_gpu'),exclusive('head_bc_v38'),awake():
        folder.mkdir();began=time.perf_counter();save(folder/'attempt.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json'),scorer_sha256=sha(__file__)))
        count=attempted=0;frozen=None;runtime=None;forward_seconds=0.
        try:
            import numpy as np,torch
            from dmpel_native_backend_v6 import NativeDMPELSource90
            from head_bc_runtime_v38 import HeadOnlyRuntime
            from check_dmpel_actor_runtime_v7 import digest_parameters
            actor=read(W/'trained_actors.json')['full_teacher_head_bc'];assert sha(actor['file'])==actor['sha256']
            native=NativeDMPELSource90('cuda',threads=2);runtime=HeadOnlyRuntime(native,77210)
            runtime.load(torch.load(actor['file'],map_location='cuda',weights_only=True)['adapter'])
            frozen=digest_parameters(runtime.frozen);assert frozen==p['frozen_pretrained_parameters_sha256']
            assert digest_parameters(list(runtime.policy.named_parameters()))==actor['all_parameter_digest']
            rows=read(W/'fixed_offline_membership.json');assert len(rows)==3291;arrays={};sums={};weights={}
            def accumulate(name,row,metrics):
                keys={(name,row['scope'],'ALL','ALL','ALL','ALL'),(name,row['scope'],row['teacher_key'],'ALL','ALL','ALL'),
                      (name,row['scope'],'ALL',row['stage'],'ALL','ALL'),(name,row['scope'],row['teacher_key'],row['stage'],row['selected_uniform'],row['selected_local'])}
                for key in keys:
                    if key not in sums:sums[key]={metric:0. for metric in metrics};weights[key]=0
                    weights[key]+=1
                    for metric,value in metrics.items():sums[key][metric]+=float(value)
            metric_names=['action_nll','nearest_mean_l2','nearest_mean_rmse','normal_sigma_min','normal_sigma_mean','normal_sigma_max','mixture_weighted_rms_sigma','mixture_entropy_nats','nearest_component_probability']
            old_counts={}
            for key,expected in [('old_fit_per_action_file',{'parent','uniform_bc','local_bc'}),('old_head_fit_per_action_file',{'head_uniform_bc'})]:
                local_counts={}
                with (ROOT/p[key]).open(encoding='utf8') as stream:
                    for line in stream:
                        old=json.loads(line);original_name=old['actor'];assert original_name in expected
                        name='head20_bc' if key=='old_head_fit_per_action_file' else original_name
                        local_counts[original_name]=local_counts.get(original_name,0)+1;old_counts[name]=old_counts.get(name,0)+1
                        assert local_counts[original_name]<=3291
                        for field,value in rows[local_counts[original_name]-1].items():assert old[field]==value
                        assert all(math.isfinite(old[metric]) for metric in metric_names)
                        accumulate(name,old,{metric:old[metric] for metric in metric_names})
                assert set(local_counts)==expected and all(n==3291 for n in local_counts.values())
            assert old_counts=={'parent':3291,'uniform_bc':3291,'local_bc':3291,'head20_bc':3291}
            queue_before=len(native.policy.latent_queue);load_done=time.perf_counter()
            with (folder/'head_per_action_fit.jsonl').open('x',encoding='utf8') as stream,torch.no_grad():
                for row in rows:
                    field=row['feature_field'];assert field in ['spatial_features','frozen_spatial_features']
                    if row['file'] not in arrays:
                        with np.load(ROOT/row['file']) as z:arrays[row['file']]=(field,z[field].copy(),z['actions'].copy())
                    actual_field,features,actions=arrays[row['file']];assert actual_field==field and features.dtype==np.float32 and features.shape[1:]==(4,768)
                    i=row['action_index'];history=runtime.history(features,i);target=torch.as_tensor(actions[i],device='cuda',dtype=torch.float32).unsqueeze(0)
                    stamp=time.perf_counter();attempted+=1;distribution=runtime.distribution(history);nll=float((-distribution.log_prob(target)).item())
                    base=distribution.component_distribution.base_dist;means=base.loc.reshape(5,7).float();sigmas=base.scale.reshape(5,7).float()
                    probs=distribution.mixture_distribution.probs.reshape(5).float();logprobs=distribution.mixture_distribution.logits.reshape(5).float()
                    errors=(means-target.reshape(1,7)).square().sum(-1).sqrt();near=int(errors.argmin().item())
                    metrics=dict(action_nll=nll,nearest_mean_l2=float(errors[near].item()),nearest_mean_rmse=float(errors[near].item())/math.sqrt(7),normal_sigma_min=float(sigmas.min().item()),normal_sigma_mean=float(sigmas.mean().item()),normal_sigma_max=float(sigmas.max().item()),mixture_weighted_rms_sigma=float((probs*sigmas.square().mean(-1)).sum().sqrt().item()),mixture_entropy_nats=float(-(probs*logprobs).sum().item()),nearest_component_probability=float(probs[near].item()))
                    assert all(math.isfinite(v) for v in metrics.values());seconds=time.perf_counter()-stamp;forward_seconds+=seconds;count+=1
                    record=dict(actor='full_teacher_head_bc',history_length=int(history.shape[1]),normalized_actual_action=actions[i].tolist(),nearest_component=near,nearest_mean=means[near].cpu().tolist(),nearest_sigma=sigmas[near].cpu().tolist(),mixture_probabilities=probs.cpu().tolist(),density_head_forward_seconds=seconds,**row,**metrics)
                    stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush();accumulate('full_teacher_head_bc',row,metrics)
            after=digest_parameters(runtime.frozen);assert after==frozen and digest_parameters(list(runtime.policy.named_parameters()))==actor['all_parameter_digest']
            assert len(native.policy.latent_queue)==queue_before and all(v.grad is None for v in runtime.parameters)
            assert count==attempted==3291
            summary=[]
            for key in sorted(sums,key=repr):
                name,scope,teacher,stage,u,l=key;summary.append(dict(actor=name,scope=scope,teacher_key=teacher,stage=stage,selected_uniform=u,selected_local=l,n=weights[key],mean={metric:value/weights[key] for metric,value in sums[key].items()}))
            verify();save(folder/'summary.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),summaries=summary,old_actor_rows_reused=old_counts,new_head_rows=count,new_head_density_forwards=3291,old_actor_model_forwards=0,encoders_or_action_samples_or_physical_controls_or_training=0,
                                                  frozen_author_before=frozen,frozen_author_after=after,new_layout_all_parameters_digest=actor['all_parameter_digest'],new_fit_file_sha256=sha(folder/'head_per_action_fit.jsonl'),old_fit_file_sha256=sha(ROOT/p['old_fit_per_action_file']),old_head_fit_file_sha256=sha(ROOT/p['old_head_fit_per_action_file']),old_total_rows_reused=13164,all_descriptive_rows=16455,
                                                  model_setup_and_old_read_seconds=load_done-began,head_density_metric_seconds=forward_seconds,within_attempt_elapsed_seconds=time.perf_counter()-began,whole_command_end_to_end_seconds=None,
                                                  limitations=['Supplementarydescriptive, correlatedteacherframesnotindependenttasks.','No adequacythreshold, pvalue, checkpointorhyperparameterselection.','Nearestcomponentmeanmaynotdominatemixturedensity; NLLnotatasksuccessprobability.','Reuseall13164oldrows; nonewoldactorforward.'],work_demonstrated=False))
            status('finite_head_baseline_complete_requires_review',completed=12,expected=12);print('Completed3291head-onlyofflinefit;0newphysics/encoder/sample.')
        except BaseException as error:
            save(folder/'failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),attempted_head_forwards=attempted,completed_head_forwards=count,elapsed_seconds=time.perf_counter()-began,physical_controls=0,automatic_retry=False));raise
if __name__=='__main__':main()
