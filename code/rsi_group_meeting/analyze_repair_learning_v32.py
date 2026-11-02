"""Audit all twenty-four original-instruction learner outcomes, then describe paired gains."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v32/repair_learning_dev';E=W/'evaluation'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
from paired_observation_v4 import array_hash
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text('utf8'))
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def main():
    assert not (W/'analysis').exists()
    p=read(W/'registration.json');assert read(E/'collection_complete.json')['physical_episodes']==24
    freeze=read(W/'analysis_frozen_before_learner_outcomes.json');assert freeze['complete_learner_outcomes_at_freeze']==0 and freeze['analysis_implementation_sha256']==sha(__file__)
    assert freeze['registration_sha256']==sha(W/'registration.json')
    for f,h in p['implementation_sha256'].items():assert sha(ROOT/f)==h
    actors=read(W/'trained_actors.json');membership=read(W/'training_membership.json');cache=read(W/'cache_complete.json')
    assert membership['registration_sha256']==cache['registration_sha256']==sha(W/'registration.json')
    for actor in actors.values():
        assert sha(actor['file'])==actor['sha256'] and actor['steps']==20 and actor['teacher_presentations']==actor['retention_presentations']==320
        assert actor['frozen_parameters_sha256']==p['frozen_pretrained_parameters_sha256']
    rows=[];initials={};arrays={}
    for episode in p['evaluation_episodes']:
        key=episode['key'];f=E/'results'/f'{key}.json';r=read(f);assert r['registration_sha256']==sha(W/'registration.json')
        for path,h in [(f.with_suffix('.npz'),'trace_sha256'),(E/r['events'],'events_sha256'),(E/r['video'],'video_sha256'),(E/r['initial_sensors'],'initial_sensors_sha256')]:assert sha(path)==r[h]
        a=load(f.with_suffix('.npz'));initial=load(E/r['initial_sensors']);events=[json.loads(line) for line in (E/r['events']).read_text('utf8').splitlines()]
        n=len(events);assert n==len(a['actions'])==r['executed_physics_steps']==r['policy_queries']
        actor=episode['actor'];expected=p['parent'] if actor=='parent' else actors[actor]
        assert r['actor_file_sha256']==r['actor_load_receipt']['sha256']==expected['sha256']
        digest=read(W/'cache_worker_metadata.json')['initial_all_parameter_digest'] if actor=='parent' else expected['all_parameter_digest']
        assert r['actor_load_receipt']['all_parameter_digest']==digest
        assert not r['privileged_controller'] and r['original_joint_instruction']
        np.testing.assert_array_equal(initial['initial_sim_state'],a['initial_sim_state'])
        for i,event in enumerate(events):
            assert event['step']==i+1 and event['history_length']==min(i+1,10)
            assert event['language']==episode['context']['setting']['language'] and event['actor']==actor
            assert np.isfinite(event['log_probability'])
            np.testing.assert_array_equal(np.asarray(event['action'],dtype=np.float32),a['actions'][i])
            assert event['goal_predicates']==a['diagnostic_subgoals'][i].tolist() and event['warning_counts']==a['native_warning_counts'][i].tolist()
            assert event['official_success']==bool(a['official_success'][i])==bool(a['diagnostic_subgoals'][i].all() and not event['instability'])
            assert event['post_state_sha256']==array_hash({'sim':a['privileged_sim_states_for_replay_only'][i]})
        assert r['success_by900']==bool(a['official_success'].any()) and r['success_by520']==bool(a['official_success'][:520].any())
        cid=episode['context']['context_id'];initials[(cid,actor)]=initial;arrays[(cid,actor)]=a
        rows.append(dict(context=cid,partition=episode['context']['partition'],actor=actor,success900=r['success_by900'],success520=r['success_by520'],instability=r['instability_terminated'],task_controls=n,inference_seconds=r['inference_seconds'],wall_seconds=r['wall_seconds'],result_sha256=sha(f)))
    paired=[]
    for c in p['evaluation_contexts']:
        cid=c['context_id'];base=initials[(cid,'parent')]
        for actor in ['uniform_bc','local_bc']:
            for field in base:np.testing.assert_array_equal(base[field],initials[(cid,actor)][field])
        outcomes={actor:next(r['success900'] for r in rows if r['context']==cid and r['actor']==actor) for actor in ['parent','uniform_bc','local_bc']}
        paired.append(dict(context=cid,partition=c['partition'],**outcomes,uniform_net=int(outcomes['uniform_bc'])-int(outcomes['parent']),local_net=int(outcomes['local_bc'])-int(outcomes['parent']),local_minus_uniform=int(outcomes['local_bc'])-int(outcomes['uniform_bc'])))
    totals={}
    for partition in ['Long_development','source_regression']:
        selected=[r for r in paired if r['partition']==partition];totals[partition]={}
        for actor in ['parent','uniform_bc','local_bc']:
            totals[partition][actor]=dict(successes=sum(r[actor] for r in selected),conditions=len(selected),repairs=sum(not r['parent'] and r[actor] for r in selected),harms=sum(r['parent'] and not r[actor] for r in selected))
    start=datetime.fromisoformat(read(E/'run_started.json')['started_utc']);end=datetime.fromisoformat(read(E/'collection_complete.json')['completed_utc'])
    prior=read(ROOT/'rsi_group_meeting/v31/handoff_factorial_dev/analysis/diagnostic_readout.json')
    costs=dict(new_training_runs=2,gradient_steps_per_run=20,teacher_presentations_per_run=320,retention_presentations_per_run=320,training_seconds=sum(a['elapsed_seconds'] for a in actors.values()),peak_torch_allocated_bytes=max(a['peak_torch_allocated_bytes'] for a in actors.values()),tensor_allocation_not_total_VRAM=True,replayed_teacher_episodes=cache['physical_replays'],replayed_controls=cache['controls'],replay_settling_controls=cache['settling_controls'],replay_spatial_encodings=cache['spatial_encodings'],replay_policy_action_queries=0,cache_wall_seconds=cache['wall_seconds'],independent_new_evaluation_episodes=24,evaluation_task_controls=sum(r['task_controls'] for r in rows),evaluation_settling_controls=240,evaluation_policy_queries=sum(r['task_controls'] for r in rows),evaluation_inference_seconds=sum(r['inference_seconds'] for r in rows),evaluation_episode_wall_sum_not_end_to_end_seconds=sum(r['wall_seconds'] for r in rows),evaluation_registered_run_elapsed_seconds=(end-start).total_seconds(),earlier_development_task_episodes_still_charged=32,earlier_v31_costs=prior['costs'],earlier_v22_and_v7_cost_ledger='rsi_group_meeting/v22/fresh_transfer/analysis/full_evaluation.json',precise_end_to_end_model_preparation_seconds=None)
    out=dict(analyzed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),rows=rows,paired_results=paired,totals=totals,costs=costs,all_initial_inputs_paired_exact=True,event_language_history_action_state_goal_warning_checks_pass=True,local_beats_uniform_in_Long_development=totals['Long_development']['local_bc']['successes']>totals['Long_development']['uniform_bc']['successes'],any_new_full_Long_repair=any(r['partition']=='Long_development' and (r['uniform_net']>0 or r['local_net']>0) for r in paired),work_demonstrated=False,limits=['A bounded two-operation feasibility check, not statistical significance or an independent paper test.','All task/initial-state families were already exposed; a new policy-noise seed does not create an unseen task.','Teachers use privileged stage timing and public servo interventions; deployed students use original joint language and public sensors only. Teacher acquisition/failures/replays/queries remain charged.','Two trained policies are two repair-process records, not eight or hundreds of meta-training tasks.','No ordinary-RL comparison or learned improver/R1/R2 experiment was performed here; even positive student outcomes do not establish innovation, matched-RL superiority or RSI.','If uniform BC works as well, prefer that baseline; do not describe local supervision as uniquely effective.'])
    folder=W/'analysis';folder.mkdir();(folder/'repair_readout.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf8')
    with (folder/'paired_results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    print(json.dumps(dict(totals=totals,any_new_full_Long_repair=out['any_new_full_Long_repair'],work_demonstrated=False)))
if __name__=='__main__':main()
