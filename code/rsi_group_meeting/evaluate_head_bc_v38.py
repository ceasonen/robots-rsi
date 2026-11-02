"""Twelve fixed free-running head-BC contexts. No teacher or GT controller."""
from pathlib import Path
import sys,os,time,json,traceback
import numpy as np
from head_bc_study_v38 import ROOT,W,read,sha,save,status,verify,now
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
def main():
    assert Path(__file__).resolve()==ROOT/'rsi_group_meeting/evaluate_head_bc_v38.py','Preparation only; root must review/copy before execution.'
    p=verify();assert read(W/'training_complete.json')['optimizer_steps']==2000
    actors=read(W/'trained_actors.json');assert set(actors)=={'full_teacher_head_bc'}
    actor=actors['full_teacher_head_bc'];assert sha(actor['file'])==actor['sha256'] and actor['steps']==2000
    assert not (W/'evaluation_started.json').exists(),'Preserve earlier evaluation; no automatic retry.'
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    from repair_learning_awake_v32 import awake
    for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='2'
    class Pipe(PolicyPipe):
        def __init__(self,*args):self.case_key=None;self.index=0;self.attempts=0;super().__init__(*args)
        def request(self,command,**args):
            if command=='query':
                packet=W/'feature_packets/cases'/self.case_key/f'{self.index:04d}.npz'
                assert not packet.exists() and not packet.with_suffix('.json').exists()
                args=dict(args,feature_packet_file=str(packet),query_case=self.case_key,query_step=self.index,query_phase='case');self.index+=1;self.attempts+=1
                with (W/'evaluation_query_receipts.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(dict(phase='attempt',ordinal=self.attempts,case=self.case_key,utc=now(),arguments=args))+'\n')
            answer=super().request(command,**args)
            if command=='query':
                with (W/'evaluation_query_receipts.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(dict(phase='completed',ordinal=self.attempts,case=self.case_key,utc=now(),answer=answer))+'\n')
            elif command in ['load_actor','reset']:
                with (W/'evaluation_policy_control_receipts.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(dict(case=self.case_key,command=command,arguments=args,answer=answer,utc=now()))+'\n')
            return answer
        def close(self):
            try:
                if self.process.poll() is None:
                    ack=self.request('close');assert ack['closed'] and ack['frozen_sha256']==p['frozen_pretrained_parameters_sha256']
                    assert ack['actor']=='full_teacher_head_bc' and ack['all_parameter_digest']==actor['all_parameter_digest']
                    save(W/'evaluation_worker_close.json',ack);self.process.wait(timeout=30)
            finally:self.terminate();self.log.close()
    with exclusive('robotics_gpu'),exclusive('head_bc_v38'),awake():
        start=time.perf_counter();save(W/'evaluation_started.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json'),pid=os.getpid()))
        status('evaluation_startup',completed=0,expected=12,current=None,worker_pid=None)
        client=None;current=None;done=0;canonical_original=None;observation_module=None
        try:
            stamp=time.perf_counter();client=Pipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/head_bc_worker_v38.py',W)
            save(W/'evaluation_worker_metadata.json',dict(metadata=client.metadata,worker_pid=client.worker_pid,startup_wall_seconds=time.perf_counter()-stamp))
            import benchmark_v4
            import evaluate_repair_learning_v32 as original
            import paired_observation_v4 as observation_module
            canonical_original=observation_module.canonicalize
            def guard(env,raw,setting,state,folder):
                measured,pair=canonical_original(env,raw,setting,state,folder);ref=current['references']['parent'];previous=read(ROOT/ref['result'])
                assert pair['model_sha256']==previous['paired_observation']['model_sha256'] and pair['full_state_sha256']==previous['paired_observation']['full_state_sha256']
                with np.load(ROOT/ref['initial_sensors']) as z:
                    for key in p['public_sensors']:np.testing.assert_array_equal(measured[key],z[key])
                save(W/'initial_input_guards'/f"{current['episode']['key']}.json",dict(checked_utc=now(),before_policy_or_task_action=True,public_arrays_equal=True,model_hash_equal=True,full_state_hash_equal=True,reference_result_sha256=sha(ROOT/ref['result'])))
                return measured,pair
            observation_module.canonicalize=guard;original.W=W;original.E=W/'evaluation'
            for case in p['cases']:
                current=case;e=case['episode'];client.case_key=e['key'];client.index=0
                status('evaluating',completed=done,expected=12,current=e['key'],partition=e['context']['partition'],worker_pid=client.worker_pid)
                per_case=dict(p);per_case['evaluation_policy_seed']=e['context']['policy_seed'];original.episode(e,per_case,client)
                r=read(W/'evaluation/results'/f"{e['key']}.json")
                with np.load(W/'evaluation'/r['initial_sensors']) as a,np.load(ROOT/case['references']['parent']['initial_sensors']) as b:
                    for key in p['public_sensors']+['initial_sim_state']:np.testing.assert_array_equal(a[key],b[key])
                    if 'initial_goal_predicates' in b:np.testing.assert_array_equal(a['initial_goal_predicates'],b['initial_goal_predicates'])
                done+=1
            observation_module.canonicalize=canonical_original;verify();client.close();client=None
            assert (W/'evaluation_worker_close.json').exists()
            save(W/'evaluation_complete.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),physical_episodes=12,settling_controls=120,within_attempt_elapsed_seconds=time.perf_counter()-start))
            status('evaluation_complete_requires_frozen_analysis',completed=12,expected=12,current=None,worker_pid=None)
        except BaseException as error:
            prefixes=[]
            if current is not None:
                # Reuse only the immutable data logger, temporarily redirected to this NEW workspace.
                import cross_context_diag_v35 as logger
                prior=logger.W
                try:logger.W=W;prefixes=logger.preserve_exception_prefix(error,current)
                finally:logger.W=prior
            save(W/'evaluation_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),completed_episodes=done,current_case=current,preserved_prefixes=prefixes,traceback=traceback.format_exc(),automatic_retry=False,infrastructure_censoring_not_task_failure=True))
            status('evaluation_stopped_requires_audit',completed=done,expected=12,current=current['episode']['key'] if current else None,worker_pid=client.worker_pid if client else None);raise
        finally:
            if observation_module is not None and canonical_original is not None:observation_module.canonicalize=canonical_original
            if client is not None:client.close()
if __name__=='__main__':main()
