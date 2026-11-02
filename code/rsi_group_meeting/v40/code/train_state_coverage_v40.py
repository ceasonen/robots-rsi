"""Two fixed BC arms differing only in retention states. No training encoder/physics."""
import sys,time,traceback
from pathlib import Path
from state_coverage_study_v40 import ROOT,W,ARMS,TRAINING_SEED,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))

def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/v40/code/train_state_coverage_v40.py','Preparation only; root must review/install.'
    import os,gc,numpy as np,torch
    from local_run_lock_v4 import exclusive
    from repair_learning_awake_v32 import awake
    p=verify();assert not (W/'training_started.json').exists(),'Preserve first attempt; never auto retry.'
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    with exclusive('robotics_gpu'),exclusive('state_coverage_v40'),awake():
        began=time.perf_counter();save(W/'training_started.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json')))
        runtime=native=optimizer=None;arm=None;step=0;current_target=None
        loss_attempted=loss_completed=forward_attempted=forward_completed=0;parity_attempted=parity_completed=0
        completed_actors={};initial_digest=None;arms_zero_states={};all_parity={}
        try:
            from dmpel_native_backend_v6 import NativeDMPELSource90
            from state_coverage_runtime_v40 import HeadOnlyRuntime
            from dmpel_policy_ports_v6 import canonical_query
            from dmpel_onpolicy_objective_v7 import gmm_joint_kl
            from check_dmpel_actor_runtime_v7 import digest_parameters
            old_state=torch.load(p['parent']['file'],map_location='cpu',weights_only=True)['adapter']
            assert len(old_state)==12
            a=[v for k,v in old_state.items() if k.endswith('lora_a')];b=[v for k,v in old_state.items() if k.endswith('lora_b')]
            assert len(a)==len(b)==6 and all(tuple(v.shape)==(4,768) and v.dtype==torch.float32 and torch.isfinite(v).all() for v in a)
            assert all(tuple(v.shape)==(2304,4) and v.dtype==torch.float32 and torch.count_nonzero(v)==0 for v in b)
            save(W/'old_parent_zero_temporal_B_proof.json',dict(file_sha256=p['parent']['sha256'],six_B_exactly_zero=True,old_temporal_adapter_loaded_into_head_layout=False))
            members=read(W/'fixed_training_membership.json');planned=read(W/'fixed_training_sampling.json');assert len(planned)==2000
            teacher_arrays={};state_arrays={};packet_arrays={}
            def teacher_target(row):
                file=ROOT/row['file'];key=str(file)
                if key not in teacher_arrays:
                    with np.load(file,allow_pickle=False) as z:teacher_arrays[key]=(z['spatial_features'].copy(),z['actions'].copy())
                f,a=teacher_arrays[key];i=row['action_index']
                assert f.dtype==np.float32 and f.shape[1:]==(4,768) and len(f)==len(a) and 0<=i<len(f)
                return f[max(0,i-9):i+1].copy(),a[i].copy()
            def retention_state(row):
                # Deliberately never access actions, proposal_action, logp or outcome fields.
                if row['kind']=='old_A_state':
                    file=ROOT/row['file'];key=str(file)
                    if key not in state_arrays:
                        with np.load(file,allow_pickle=False) as z:state_arrays[key]=z['frozen_spatial_features'].copy()
                    f=state_arrays[key];i=row['action_index'];assert 9<=i<len(f)
                    history=f[i-9:i+1].copy()
                else:
                    assert row['kind']=='cached_source_state' and len(row['history_packets'])==10
                    f=[]
                    for packet in row['history_packets']:
                        file=ROOT/packet['file'];key=str(file)
                        if key not in packet_arrays:
                            assert sha(file)==packet['sha256']
                            with np.load(file,allow_pickle=False) as z:packet_arrays[key]=z['spatial_feature'].copy()
                        f.append(packet_arrays[key])
                    history=np.stack(f)
                assert history.shape==(10,4,768) and history.dtype==np.float32 and np.isfinite(history).all()
                return history
            assert len(members['full_teacher'])==3163
            for row in members['full_teacher']:teacher_target(row)
            for which in ARMS:
                assert len(members['retention_by_arm'][which])==128
                for row in members['retention_by_arm'][which]:retention_state(row)
            for arm in ARMS:
                step=0;loss_attempted=loss_completed=forward_attempted=forward_completed=0;parity_attempted=parity_completed=0
                status('training_arm_startup',arm=arm,step=0,completed_arms=list(completed_actors))
                save(W/'training_attempts'/f'{arm}.json',dict(started_utc=now(),arm=arm,initialization='Fresh authorpi0 +zeroB, not previousarm.',registration_sha256=sha(W/'registration.json')))
                loading=time.perf_counter();native=NativeDMPELSource90('cuda',threads=2)
                original_parameters=list(native.policy.named_parameters());frozen=digest_parameters(original_parameters)
                assert frozen==p['frozen_pretrained_parameters_sha256'];native.policy.policy_head.low_eval_noise=False
                save(W/'arms'/arm/'native_metadata.json',dict(metadata=native.metadata,native_construction_wall_seconds=time.perf_counter()-loading,author_named_parameter_sha256=frozen))
                def probe(phase,index,item):
                    nonlocal parity_attempted,parity_completed
                    save(W/'arms'/arm/'parity_attempts'/phase/f'{index:02d}.json',dict(started_utc=now(),context=item['context'],physical_controls=0))
                    seen={}
                    def hook(module,inputs,output):seen['output']=output.detach().cpu().numpy().copy();seen['dtype']=str(output.dtype)
                    handle=native.policy.policy_head.share[0].register_forward_hook(hook)
                    try:
                        native.reset(item['context']['policy_seed'])
                        with np.load(item['observation_file'],allow_pickle=False) as z:raw={key:z[key] for key in p['public_sensors']}
                        stamp=time.perf_counter();parity_attempted+=1
                        action,logp,feature,distribution=canonical_query(native,raw,item['context']['setting']['language'])
                        assert seen['dtype']=='torch.float16';normal=distribution.component_distribution.base_dist
                        values=dict(action=action,log_probability=np.asarray(logp,dtype=np.float64),feature=feature,
                          means=normal.loc.detach().cpu().numpy(),scales=normal.scale.detach().cpu().numpy(),
                          logits=distribution.mixture_distribution.logits.detach().cpu().numpy(),first_linear_output=seen['output'],first_linear_dtype=np.asarray(seen['dtype']))
                        elapsed=time.perf_counter()-stamp;parity_completed+=1
                        file=W/'arms'/arm/'parity_records'/phase/f'{index:02d}.npz';file.parent.mkdir(parents=True,exist_ok=True)
                        with file.open('xb') as stream:np.savez_compressed(stream,**values)
                        save(file.with_suffix('.json'),dict(arm=arm,phase=phase,index=index,file_sha256=sha(file),query_seconds=elapsed,query_encoders=1,query_heads=1,action_samples=1,physical_controls=0))
                        return values
                    finally:handle.remove()
                plain=[]
                for i,item in enumerate(p['parity']):
                    status('plain_parent_parity',arm=arm,index=i,step=0);record=probe('plain_author',i,item);plain.append(record)
                    np.testing.assert_array_equal(record['action'],np.asarray(item['expected_action'],dtype=np.float32))
                    assert abs(float(record['log_probability'])-item['expected_log_probability'])<=item['legacy_logp_abs_tolerance']
                assert digest_parameters(original_parameters)==frozen
                runtime=HeadOnlyRuntime(native,77210);assert digest_parameters(runtime.frozen)==frozen
                assert torch.count_nonzero(runtime.layer.lora_b)==0 and runtime.layer.scale==2.
                zero_digest=digest_parameters(list(runtime.policy.named_parameters()))
                if initial_digest is None:initial_digest=zero_digest
                else:assert initial_digest==zero_digest,'Both arms must start with identical allparameter layout/values.'
                zero_state=runtime.state()
                if arms_zero_states:
                    previous=next(iter(arms_zero_states.values()))
                    assert set(previous)==set(zero_state)
                    for name in previous:assert torch.equal(previous[name],zero_state[name])
                arms_zero_states[arm]=zero_state
                save(W/'arms'/arm/'initial_identity.json',dict(initial_all_parameter_digest=zero_digest,author_named_parameter_sha256=frozen,
                    zero_B=True,A_seed=77210,trainable_parameters=73728,reference='authorpi0'))
                for i,item in enumerate(p['parity']):
                    status('zero_head_parity',arm=arm,index=i,step=0);record=probe('zero_head',i,item)
                    for field in plain[i]:assert record[field].dtype==plain[i][field].dtype;np.testing.assert_array_equal(record[field],plain[i][field])
                assert parity_attempted==parity_completed==8 and digest_parameters(runtime.frozen)==frozen
                parity_receipt=dict(completed_utc=now(),arm=arm,total_queries=8,total_encoders=8,total_action_samples=8,physical_controls=0,
                  plain_queries=4,head_zero_queries=4,all_GMM_action_feature_first_linear_values_and_dtype_exact=True,initial_all_parameter_digest=zero_digest,
                  author_named_parameter_sha256=frozen,legacy_logp_abs_tolerance=1e-5)
                save(W/'arms'/arm/'zero_B_functional_parity_complete.json',parity_receipt);all_parity[arm]=parity_receipt
                torch.manual_seed(TRAINING_SEED);rng=np.random.default_rng(TRAINING_SEED)
                optimizer=torch.optim.AdamW(runtime.parameters,lr=1e-4,weight_decay=0,betas=(.9,.999),eps=1e-8)
                torch.cuda.reset_peak_memory_stats();metrics=[];sampling=[];training_began=time.perf_counter()
                for step in range(1,2001):
                    status('training',arm=arm,step=step,completed_arms=list(completed_actors));optimizer.zero_grad(set_to_none=True)
                    loss_sum=teacher_nll=teacher_kl=retention_kl=0.;draws={}
                    for pool in ['full_teacher','retention']:
                        entries=members['full_teacher'] if pool=='full_teacher' else members['retention_by_arm'][arm]
                        indices=rng.integers(0,len(entries),size=16).tolist();assert indices==planned[step-1][pool];draws[pool]=indices
                        for j in indices:
                            current_target=dict(arm=arm,step=step,pool=pool,selected_pool_index=j,row=entries[j]);loss_attempted+=1
                            if pool=='full_teacher':features,action=teacher_target(entries[j])
                            else:features=retention_state(entries[j])
                            history=torch.from_numpy(features).unsqueeze(0).to('cuda');forward_attempted+=1
                            with torch.no_grad(),runtime.reference():reference=runtime.distribution(history)
                            forward_completed+=1;forward_attempted+=1;current=runtime.distribution(history);forward_completed+=1
                            kl=gmm_joint_kl(current,reference)
                            if pool=='full_teacher':
                                target=torch.as_tensor(action,device='cuda',dtype=torch.float32).unsqueeze(0)
                                nll=-current.log_prob(target);loss=(nll+.001*kl).sum()/32
                                teacher_nll+=float(nll.detach().sum().cpu())/16;teacher_kl+=float(kl.detach().sum().cpu())/16
                            else:
                                loss=kl.sum()/32;retention_kl+=float(kl.detach().sum().cpu())/16
                            assert torch.isfinite(loss) and loss.requires_grad;loss.backward();loss_completed+=1;loss_sum+=float(loss.detach().cpu())
                    assert all(v.grad is None for _,v in runtime.frozen)
                    assert all(v.dtype==torch.float32 and v.grad is not None and v.grad.dtype==torch.float32 and torch.isfinite(v.grad).all() for v in runtime.parameters)
                    nonzero=[int(torch.count_nonzero(v.grad).cpu()) for v in runtime.parameters]
                    norm=torch.nn.utils.clip_grad_norm_(runtime.parameters,1.);assert torch.isfinite(norm);optimizer.step()
                    assert all(torch.isfinite(v).all() for v in runtime.parameters)
                    row=dict(arm=arm,step=step,loss=loss_sum,teacher_nll=teacher_nll,teacher_current_to_parent_joint_kl=teacher_kl,
                      retention_current_to_parent_joint_kl=retention_kl,teacher_joint_kl_coefficient=.001,retention_joint_kl_coefficient=1.,
                      gradient_norm_before_clip=float(norm.cpu()),nonzero_gradient_entries_A_B=nonzero,gradient_dtype='float32',teacher_presentations=16,
                      retention_state_presentations=16,retention_action_labels_read=0,completed_loss_presentations=loss_completed,completed_distribution_forwards=forward_completed)
                    metrics.append(row);sampling.append(dict(step=step,**draws))
                    save(W/'arms'/arm/'training_metrics'/f'step{step:04d}.json',row);save(W/'arms'/arm/'training_sampling_steps'/f'step{step:04d}.json',sampling[-1])
                assert loss_attempted==loss_completed==64000 and forward_attempted==forward_completed==128000 and sampling==planned
                optimizer.zero_grad(set_to_none=True);state=runtime.state();assert any(not torch.equal(state[k],zero_state[k]) for k in state)
                assert digest_parameters(runtime.frozen)==frozen
                file=W/'actors'/f'{arm}_step2000.pt';file.parent.mkdir(parents=True,exist_ok=True);assert not file.exists()
                torch.save(dict(adapter=state,adapter_kind='head_input_rank72_alpha144'),file)
                effect=float(torch.linalg.vector_norm(runtime.layer.lora_b.detach()@runtime.layer.lora_a.detach()*runtime.layer.scale).cpu())
                actor=dict(file=str(file),sha256=sha(file),steps=2000,adapter_kind='head_input_rank72_alpha144',trainable_parameters=73728,
                  teacher_presentations=32000,retention_state_presentations=32000,retention_action_labels_read=0,
                  completed_loss_presentations=loss_completed,completed_distribution_forwards=forward_completed,
                  all_parameter_digest=digest_parameters(list(runtime.policy.named_parameters())),initial_all_parameter_digest=zero_digest,
                  frozen_parameters_sha256=frozen,training_loop_wall_seconds=time.perf_counter()-training_began,
                  peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),effective_head_delta_weight_Frobenius_norm=effect,
                  sampling_sha256=sha(W/'fixed_training_sampling.json'),metrics=metrics)
                save(W/'arms'/arm/'training_sampling.json',sampling);save(W/'arms'/arm/'training_receipt.json',actor)
                completed_actors[arm]=actor
                save(W/'arms'/arm/'training_complete.json',dict(completed_utc=now(),arm=arm,steps=2000,completed_loss_presentations=64000,
                  completed_distribution_forwards=128000,initial_all_parameter_digest=zero_digest,frozen_parameters_sha256=frozen,parity_queries=8))
                # Free the first independent model before constructing the second; no arm continues another.
                optimizer=None;runtime=None;native=None;original_parameters=None
                current=reference=history=target=loss=kl=None
                gc.collect();torch.cuda.empty_cache()
            assert set(completed_actors)==set(ARMS) and len(all_parity)==2
            save(W/'trained_actors.json',completed_actors)
            save(W/'zero_B_functional_parity_complete.json',dict(completed_utc=now(),arms=all_parity,total_queries=16,total_encoders=16,total_action_samples=16,
              physical_controls=0,initial_all_parameter_digest=initial_digest,both_initial_layout_parameter_digests_equal=True))
            receipt=dict(completed_utc=now(),arms=completed_actors,initial_all_parameter_digest=initial_digest,sampling_sha256=sha(W/'fixed_training_sampling.json'),
              training_runs=2,optimizer_steps_per_arm=2000,total_optimizer_steps=4000,distribution_forwards_per_arm=128000,total_distribution_forwards=256000,
              total_teacher_presentations=64000,total_retention_state_presentations=64000,retention_action_labels_read=0,parity_queries=16,
              within_attempt_elapsed_seconds=time.perf_counter()-began,precise_whole_command_wall_seconds=None)
            save(W/'training_receipt.json',receipt);verify()
            save(W/'training_complete.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),optimizer_steps_per_arm=2000,
              total_optimizer_steps=4000,training_runs=2,total_distribution_forwards=256000,total_loss_presentations=128000,
              total_teacher_presentations=64000,total_retention_state_presentations=64000,retention_action_labels_read=0,parity_queries=16,
              initial_all_parameter_digest=initial_digest,within_attempt_elapsed_seconds=time.perf_counter()-began,precise_end_to_end_command_wall_seconds=None))
            status('training_complete_requires_evaluation',completed_arms=ARMS,step=2000)
        except BaseException as error:
            tensor_error=optimizer_error=None
            if optimizer is not None:
                try:
                    file=W/'failures'/str(arm)/'optimizer_state_not_candidate.pt';file.parent.mkdir(parents=True,exist_ok=True);assert not file.exists();torch.save(optimizer.state_dict(),file)
                except BaseException as capture:optimizer_error=repr(capture)
            if runtime is not None:
                try:
                    values={}
                    for name,value in [('A',runtime.layer.lora_a),('B',runtime.layer.lora_b)]:
                        values[name]=value.detach().cpu().numpy()
                        if value.grad is not None:values[name+'_grad']=value.grad.detach().cpu().numpy()
                    file=W/'failures'/str(arm)/'tensors_not_candidate.npz';file.parent.mkdir(parents=True,exist_ok=True)
                    with file.open('xb') as stream:np.savez_compressed(stream,**values)
                except BaseException as capture:tensor_error=repr(capture)
            failure=dict(stopped_utc=now(),error=repr(error),traceback=traceback.format_exc(),arm=arm,step=step,current_target=current_target,
              completed_arms=list(completed_actors),loss_presentations_attempted=loss_attempted,loss_presentations_completed=loss_completed,
              distribution_forwards_attempted=forward_attempted,distribution_forwards_completed=forward_completed,
              parity_attempted_current_arm=parity_attempted,parity_completed_current_arm=parity_completed,
              tensor_preservation_error=tensor_error,optimizer_preservation_error=optimizer_error,
              within_attempt_elapsed_seconds=time.perf_counter()-began,physical_controls=0,auto_retry_allowed=False)
            save(W/'training_failure.json',failure);status('stopped_requires_audit',arm=arm,step=step,completed_arms=list(completed_actors));raise
        finally:
            optimizer=None;runtime=None;native=None;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
