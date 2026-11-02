"""One fixed standard BC run. No physical rollout or new teacher collection."""
import sys,time,math,traceback
from pathlib import Path
from head_bc_study_v38 import ROOT,W,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'scripts'))

def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/train_head_bc_v38.py','Preparation only; root must review/copy before execution.'
    import os,numpy as np,torch
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    p=verify();assert not (W/'training_started.json').exists(),'Preserve the first training attempt.'
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    with exclusive('robotics_gpu'),exclusive('head_bc_v38'),awake():
        began=time.perf_counter();save(W/'training_started.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json')))
        status('training_startup',step=0);runtime=None;native=None;attempted=completed=0;step=0;current_target=None;loss_attempted=loss_completed=forward_attempted=forward_completed=0
        try:
            from dmpel_native_backend_v6 import NativeDMPELSource90
            from head_bc_runtime_v38 import HeadOnlyRuntime
            from dmpel_policy_ports_v6 import canonical_query
            from dmpel_onpolicy_objective_v7 import gmm_joint_kl
            from check_dmpel_actor_runtime_v7 import digest_parameters
            # This tiny old file is a logical-parent identity proof, never loaded into the new layout.
            old_state=torch.load(p['parent']['file'],map_location='cpu',weights_only=True)['adapter']
            assert len(old_state)==12
            old_a=[v for k,v in old_state.items() if k.endswith('lora_a')];old_b=[v for k,v in old_state.items() if k.endswith('lora_b')]
            assert len(old_a)==len(old_b)==6
            assert all(tuple(v.shape)==(4,768) and v.dtype==torch.float32 and torch.isfinite(v).all() for v in old_a)
            assert all(tuple(v.shape)==(2304,4) and v.dtype==torch.float32 and torch.count_nonzero(v)==0 for v in old_b)
            save(W/'old_parent_zero_temporal_B_proof.json',dict(file_sha256=p['parent']['sha256'],six_B_exactly_zero=True,old_adapter_loaded_into_new_head=False))
            loading=time.perf_counter();native=NativeDMPELSource90('cuda',threads=2)
            original_parameters=list(native.policy.named_parameters());frozen=digest_parameters(original_parameters)
            assert frozen==p['frozen_pretrained_parameters_sha256'];native.policy.policy_head.low_eval_noise=False
            save(W/'native_metadata.json',dict(metadata=native.metadata,native_construction_wall_seconds=time.perf_counter()-loading,author_named_parameter_sha256=frozen))
            def probe(phase,index,item):
                nonlocal attempted,completed
                save(W/'parity_attempts'/phase/f'{index:02d}.json',dict(started_utc=now(),context=item['context'],physical_controls=0))
                seen={}
                def hook(module,inputs,output):seen['output']=output.detach().cpu().numpy().copy();seen['dtype']=str(output.dtype)
                handle=native.policy.policy_head.share[0].register_forward_hook(hook)
                try:
                    native.reset(item['context']['policy_seed'])
                    with np.load(item['observation_file']) as z:raw={key:z[key] for key in p['public_sensors']}
                    stamp=time.perf_counter();attempted+=1
                    action,logp,feature,distribution=canonical_query(native,raw,item['context']['setting']['language'])
                    assert seen['dtype']=='torch.float16'
                    normal=distribution.component_distribution.base_dist
                    values=dict(action=action,log_probability=np.asarray(logp,dtype=np.float64),feature=feature,
                                means=normal.loc.detach().cpu().numpy(),scales=normal.scale.detach().cpu().numpy(),
                                logits=distribution.mixture_distribution.logits.detach().cpu().numpy(),
                                first_linear_output=seen['output'],first_linear_dtype=np.asarray(seen['dtype']))
                    elapsed=time.perf_counter()-stamp;completed+=1
                    file=W/'parity_records'/phase/f'{index:02d}.npz';file.parent.mkdir(parents=True,exist_ok=True)
                    with file.open('xb') as stream:np.savez_compressed(stream,**values)
                    save(file.with_suffix('.json'),dict(phase=phase,index=index,file_sha256=sha(file),query_seconds=elapsed,query_encoders=1,query_heads=1,action_samples=1,physical_controls=0))
                    return values
                finally:handle.remove()
            plain=[]
            for i,item in enumerate(p['parity']):
                status('plain_parent_parity',index=i,step=0);record=probe('plain_author',i,item);plain.append(record)
                np.testing.assert_array_equal(record['action'],np.asarray(item['expected_action'],dtype=np.float32))
                assert abs(float(record['log_probability'])-item['expected_log_probability'])<=item['legacy_logp_abs_tolerance']
            assert digest_parameters(original_parameters)==frozen
            runtime=HeadOnlyRuntime(native,77210);assert digest_parameters(runtime.frozen)==frozen
            assert torch.count_nonzero(runtime.layer.lora_b)==0 and runtime.layer.scale==2.
            for i,item in enumerate(p['parity']):
                status('zero_head_parity',index=i,step=0);record=probe('zero_head',i,item)
                for field in plain[i]:
                    assert record[field].dtype==plain[i][field].dtype
                    np.testing.assert_array_equal(record[field],plain[i][field])
            assert attempted==completed==8 and digest_parameters(runtime.frozen)==frozen
            save(W/'zero_B_functional_parity_complete.json',dict(completed_utc=now(),plain_queries=4,head_zero_queries=4,total_queries=8,total_encoders=8,total_action_samples=8,physical_controls=0,all_GMM_action_feature_first_linear_values_and_dtype_exact=True,legacy_logp_tolerance=1e-5,author_named_parameter_sha256=frozen))
            members=read(W/'fixed_training_membership.json');arrays={}
            def sample(row):
                file=ROOT/row['file'];field=row['feature_field'];assert field in ['spatial_features','frozen_spatial_features']
                if str(file) not in arrays:
                    with np.load(file) as z:arrays[str(file)]=(field,z[field].copy(),z['actions'].copy())
                actual_field,features,actions=arrays[str(file)];index=row['action_index'];assert actual_field==field and 0<=index<len(features)==len(actions) and features.dtype==np.float32 and features.shape[1:]==(4,768)
                history=features[max(0,index-9):index+1].copy();assert len(history)==min(index+1,10)
                return history,actions[index].copy()
            for pool in ['full_teacher','retention']:
                assert len(members['selected_targets'][pool])==(3163 if pool=='full_teacher' else 128)
                for row in members['selected_targets'][pool]:sample(row)
            planned=read(W/'fixed_training_sampling.json');assert len(planned)==2000
            torch.manual_seed(20265450);rng=np.random.default_rng(20265450)
            optimizer=torch.optim.AdamW(runtime.parameters,lr=1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
            torch.cuda.reset_peak_memory_stats();initial=runtime.state();metrics=[];sampling=[];training_began=time.perf_counter()
            for step in range(1,2001):
                status('training',step=step);optimizer.zero_grad(set_to_none=True);loss_sum=kl_sum=teacher_nll=retention_nll=0.;draws={}
                for pool in ['full_teacher','retention']:
                    entries=members['selected_targets'][pool];indices=rng.integers(0,len(entries),size=16).tolist();assert indices==planned[step-1][pool];draws[pool]=indices
                    for j in indices:
                        current_target=dict(step=step,pool=pool,selected_pool_index=j,row=entries[j]);features,action=sample(entries[j])
                        loss_attempted+=1
                        history=torch.from_numpy(features).unsqueeze(0).to('cuda');target=torch.as_tensor(action,device='cuda',dtype=torch.float32).unsqueeze(0)
                        forward_attempted+=1
                        with torch.no_grad(),runtime.reference():reference=runtime.distribution(history)
                        forward_completed+=1;forward_attempted+=1
                        current=runtime.distribution(history);forward_completed+=1;nll=-current.log_prob(target);kl=gmm_joint_kl(current,reference);loss=(nll+.001*kl).sum()/32
                        assert torch.isfinite(loss) and loss.requires_grad;loss.backward();loss_completed+=1;loss_sum+=float(loss.detach().cpu());kl_sum+=float(kl.detach().sum().cpu())/32
                        if pool=='full_teacher':teacher_nll+=float(nll.detach().sum().cpu())/16
                        else:retention_nll+=float(nll.detach().sum().cpu())/16
                assert all(v.grad is None for _,v in runtime.frozen)
                assert all(v.dtype==torch.float32 and v.grad is not None and v.grad.dtype==torch.float32 and torch.isfinite(v.grad).all() for v in runtime.parameters)
                nonzero=[int(torch.count_nonzero(v.grad).cpu()) for v in runtime.parameters]
                norm=torch.nn.utils.clip_grad_norm_(runtime.parameters,1.);assert torch.isfinite(norm);optimizer.step()
                assert all(torch.isfinite(v).all() for v in runtime.parameters)
                row=dict(step=step,loss=loss_sum,joint_kl_upper_bound=kl_sum,teacher_nll=teacher_nll,retention_nll=retention_nll,gradient_norm_before_clip=float(norm.cpu()),nonzero_gradient_entries_A_B=nonzero,gradient_dtype='float32',teacher_presentations=16,retention_presentations=16,completed_loss_presentations=loss_completed,completed_distribution_forwards=forward_completed)
                metrics.append(row);sampling.append(dict(step=step,**draws));save(W/'training_metrics'/f'step{step:04d}.json',row);save(W/'training_sampling_steps'/f'step{step:04d}.json',sampling[-1])
            assert loss_attempted==loss_completed==64000 and forward_attempted==forward_completed==128000
            assert sampling==planned
            optimizer.zero_grad(set_to_none=True);state=runtime.state();assert any(not torch.equal(state[k],initial[k]) for k in state)
            assert digest_parameters(runtime.frozen)==frozen
            file=W/'actors/full_teacher_head_step2000.pt';file.parent.mkdir(parents=True,exist_ok=True);assert not file.exists()
            torch.save(dict(adapter=state,adapter_kind='head_input_rank72_alpha144'),file)
            effect=float(torch.linalg.vector_norm(runtime.layer.lora_b.detach()@runtime.layer.lora_a.detach()*runtime.layer.scale).cpu())
            actor=dict(file=str(file),sha256=sha(file),steps=2000,adapter_kind='head_input_rank72_alpha144',trainable_parameters=73728,
                       teacher_presentations=32000,retention_presentations=32000,teacher_pool_size=3163,retention_pool_size=128,all_3163_teacher_targets_presented=True,all_parameter_digest=digest_parameters(list(runtime.policy.named_parameters())),
                       frozen_parameters_sha256=frozen,training_loop_wall_seconds=time.perf_counter()-training_began,peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
                       effective_head_delta_weight_Frobenius_norm=effect,metrics=metrics)
            save(W/'training_sampling.json',sampling);save(W/'training_receipt.json',actor);save(W/'trained_actors.json',{'full_teacher_head_bc':actor})
            verify();save(W/'training_complete.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),new_training_runs=1,optimizer_steps=2000,loss_presentations=64000,training_distribution_forwards=128000,parity_queries=8,within_attempt_elapsed_seconds=time.perf_counter()-began,precise_end_to_end_command_wall_seconds=None))
            status('training_complete_requires_evaluation',step=2000)
        except BaseException as error:
            capture_error=None;optimizer_preservation_error=None
            if 'optimizer' in locals():
                try:
                    file=W/'failure_optimizer_state_not_candidate.pt';assert not file.exists();torch.save(optimizer.state_dict(),file)
                except BaseException as capture:optimizer_preservation_error=repr(capture)
            if runtime is not None:
                try:
                    values={}
                    for name,value in [('A',runtime.layer.lora_a),('B',runtime.layer.lora_b)]:
                        values[name]=value.detach().cpu().numpy()
                        if value.grad is not None:values[name+'_grad']=value.grad.detach().cpu().numpy()
                    file=W/'failure_training_tensors_not_candidate.npz'
                    with file.open('xb') as stream:np.savez_compressed(stream,**values)
                except BaseException as capture:capture_error=repr(capture)
            save(W/'training_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),tensor_preservation_error=capture_error,optimizer_preservation_error=optimizer_preservation_error,loss_presentations_attempted=loss_attempted,loss_presentations_completed=loss_completed,distribution_forwards_attempted=forward_attempted,distribution_forwards_completed=forward_completed,step=step,current_target=current_target,parity_queries_attempted=attempted,parity_queries_completed=completed,elapsed_seconds=time.perf_counter()-began,physical_controls=0,automatic_retry=False))
            status('training_stopped_requires_audit',step=step);raise
if __name__=='__main__':main()
