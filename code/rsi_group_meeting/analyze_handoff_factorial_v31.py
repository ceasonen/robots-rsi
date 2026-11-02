"""Frozen descriptive analysis and execution audit of the twelve-case diagnosis."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,csv,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
W=ROOT/'rsi_group_meeting/v31/handoff_factorial_dev'
OLD=ROOT/'rsi_group_meeting/v30/early_handoff_dev'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
from paired_observation_v4 import array_hash
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p) as z:return {k:z[k].copy() for k in z.files}
def trace(folder,e):
    f=folder/'results'/f"{e['key']}.json";r=read(f)
    assert r['registration_sha256']==sha(folder/'registration.json')
    for path,key in [(f.with_suffix('.npz'),'trace_sha256'),(folder/r['events'],'events_sha256'),(folder/r['video'],'video_sha256'),(folder/r['initial_sensors'],'initial_sensors_sha256')]:assert sha(path)==r[key]
    return r,load(f.with_suffix('.npz')),load(folder/r['initial_sensors']),[json.loads(x) for x in (folder/r['events']).read_text('utf8').splitlines()]
def servo_expected(info):
    pos=np.asarray(info['public_robot_site_position']);rot=np.asarray(info['public_robot_site_rotation']);target=np.asarray(info['target_rotation'])
    err=.5*sum((np.cross(rot[:,i],target[:,i]) for i in range(3)))
    output=np.r_[.5*(np.asarray(info['target_position'])-pos),.5*err]
    imin=np.asarray(info['input_min']);imax=np.asarray(info['input_max']);omin=np.asarray(info['output_min']);omax=np.asarray(info['output_max'])
    action=(np.clip(output,omin,omax)-(omax+omin)/2)/(np.abs(omax-omin)/np.abs(imax-imin))+(imax+imin)/2
    return np.r_[np.clip(action,imin,imax),-1.].astype(np.float32)
def main():
    assert not (W/'analysis').exists(),'Preserve an existing analysis; audit before any retry.'
    done=read(W/'collection_complete.json');assert done['physical_episodes']==12
    freeze=read(W/'analysis_frozen_before_development_outcomes.json')
    assert sha(__file__)==freeze['analysis_implementation_sha256'] and freeze['complete_outcomes_at_freeze']==0
    p=read(W/'registration.json');assert sha(W/'registration.json')==freeze['registration_sha256']
    for f,h in p['implementation_sha256'].items():assert sha(ROOT/f)==h
    for f,h in freeze['reused_control_and_preflight_files_sha256'].items():assert sha(ROOT/f)==h
    assert read(W/'worker_metadata.json')['all_parameter_digest']==read(OLD/'worker_metadata.json')['all_parameter_digest']
    controls={}
    for e in read(OLD/'registration.json')['episodes']:controls[e['context']['context_id']]=trace(OLD,e)
    rows=[];arrays={}
    for e in p['episodes']:
        r,a,initial,events=trace(W,e);c=e['context'];prior,b,oldinitial,_=controls[c['context_id']]
        for name in initial:np.testing.assert_array_equal(initial[name],oldinitial[name])
        n=len(events);assert n==len(a['actions'])==r['executed_physics_steps']==r['policy_queries']
        np.testing.assert_array_equal(a['initial_sim_state'],initial['initial_sim_state'])
        stages=[];ticks=0;trigger_pos=None;home=np.asarray(r['home_position']);home_rot=np.asarray(r['home_rotation'])
        for i,event in enumerate(events):
            stage=int(i>0 and a['diagnostic_subgoals'][:i,0].any());stages.append(stage)
            mode='author_low_eval_noise' if stage and e['low_noise'] else 'learned'
            is_home=bool(stage and e['home'] and ticks<60)
            assert event['step']==i+1 and event['history_length']==min(i+1,10)
            assert event['language']==c['subcommands'][stage]
            assert event['decode_mode']==mode==a['decoder_modes'][i]
            assert event['action_source']==('servo' if is_home else 'policy')==a['action_sources'][i]
            np.testing.assert_array_equal(np.asarray(event['action'],dtype=np.float32),a['actions'][i])
            np.testing.assert_array_equal(np.asarray(event['proposal_action'],dtype=np.float32),a['proposal_actions'][i])
            assert event['proposal_log_probability']==float(a['proposal_log_probabilities'][i])
            if is_home:
                info=event['servo_proprioception'];assert event['behavior_log_probability'] is None
                if trigger_pos is None:trigger_pos=np.asarray(info['public_robot_site_position'])
                z=max(home[2],trigger_pos[2]+.12)
                expected_target=np.r_[trigger_pos[:2],z] if ticks<20 else np.r_[home[:2],z] if ticks<40 else home
                np.testing.assert_array_equal(np.asarray(info['target_position']),expected_target)
                np.testing.assert_array_equal(np.asarray(info['target_rotation']),home_rot)
                np.testing.assert_allclose(a['actions'][i],servo_expected(info),rtol=0,atol=1e-7)
                ticks+=1
            else:
                assert event['servo_proprioception'] is None
                assert event['behavior_log_probability']==event['proposal_log_probability']
                np.testing.assert_array_equal(a['actions'][i],a['proposal_actions'][i])
            if mode=='author_low_eval_noise':
                assert abs(event['normal_scale_min']-1e-4)<1e-7 and abs(event['normal_scale_max']-1e-4)<1e-7
            assert event['normal_scale_min']>0 and event['normal_scale_max']>=event['normal_scale_min']
            assert event['goal_predicates']==a['diagnostic_subgoals'][i].tolist()
            assert event['warning_counts']==a['native_warning_counts'][i].tolist()
            assert event['official_success']==bool(a['official_success'][i])==bool(a['diagnostic_subgoals'][i].all() and not event['instability'])
            assert event['post_state_sha256']==array_hash({'sim':a['privileged_sim_states_for_replay_only'][i]})
        assert ticks==r['servo_controls']
        assert r['success_by900']==bool(a['official_success'].any()) and r['success_by520']==bool(a['official_success'][:520].any())
        np.testing.assert_allclose(sum(x['query_seconds'] for x in events),r['inference_seconds'],rtol=1e-12)
        switch=np.flatnonzero(stages);prefix=int(switch[0]) if len(switch) else n
        ae=float(np.abs(a['actions'][:prefix]-b['actions'][:prefix]).max()) if prefix else 0.
        se=float(np.abs(a['privileged_sim_states_for_replay_only'][:prefix]-b['privileged_sim_states_for_replay_only'][:prefix]).max()) if prefix else 0.
        stable=n-int(r['instability_terminated']);diff=int(r['success_by900'])-int(prior['success_by900'])
        row=dict(context=c['context_id'],arm=e['arm'],success_by900=r['success_by900'],success_by520=r['success_by520'],reused_nohome_learned_success=prior['success_by900'],paired_difference=diff,switch_action=prefix+1 if len(switch) else None,first_goal_retained=bool(stable and a['diagnostic_subgoals'][stable-1,0]),second_goal_ever_stably_true=bool(a['diagnostic_subgoals'][:stable,1].any()),instability=r['instability_terminated'],servo_controls=ticks,home_exit_position_error_m=r['home_exit_position_error_m'],terminal_EE_distance_to_home_m=r['terminal_EE_distance_to_home_m'],prefix_max_action_error=ae,prefix_max_fullstate_error=se,prefix_exact=ae==se==0,task_controls=n,inference_seconds=r['inference_seconds'],wall_seconds=r['wall_seconds'],result_sha256=sha(W/'results'/f"{e['key']}.json"))
        rows.append(row);arrays[(c['context_id'],e['arm'])]=(a,prefix,ticks)
    pairs=[]
    for context in controls:
        a,prefix,ta=arrays[(context,'home_learned')];b,pb,tb=arrays[(context,'home_low')]
        stop=min(prefix+ta,pb+tb);ae=float(np.abs(a['actions'][:stop]-b['actions'][:stop]).max());se=float(np.abs(a['privileged_sim_states_for_replay_only'][:stop]-b['privileged_sim_states_for_replay_only'][:stop]).max())
        pairs.append(dict(context=context,compared_controls=stop,same_switch_and_servo_lengths=prefix==pb and ta==tb,action_error=ae,fullstate_error=se,exact=prefix==pb and ta==tb and ae==se==0))
    cells={}
    for arm in ['nohome_learned','home_learned','nohome_low','home_low']:
        these=[r for r in rows if r['arm']==arm]
        cells[arm]=dict(successes=sum(x[0]['success_by900'] for x in controls.values()) if arm=='nohome_learned' else sum(r['success_by900'] for r in these),episodes=4,new_episodes=0 if arm=='nohome_learned' else 4,repairs=None if not these else sum(r['paired_difference']>0 for r in these),harms=None if not these else sum(r['paired_difference']<0 for r in these))
    start=datetime.fromisoformat(read(W/'run_started.json')['started_utc']);end=datetime.fromisoformat(done['completed_utc'])
    pre=read(W/'servo_preflight_receipt.json');assert pre['passed'] and pre['registration_sha256']==sha(W/'registration.json')
    out=dict(analyzed_utc=datetime.now(timezone.utc).isoformat(),registration_sha256=sha(W/'registration.json'),new_physical_episodes=12,reused_physical_episodes=4,semantic_task_families=2,cells=cells,rows=rows,all_A_prefixes_exact=all(r['prefix_exact'] for r in rows),home_learned_vs_home_low_common_physical_path=pairs,descriptive_interaction_success_count=cells['home_low']['successes']-cells['home_learned']['successes']-cells['nohome_low']['successes']+cells['nohome_learned']['successes'],branch_rejected_if_all_new_arms_fail=not any(r['success_by900'] for r in rows),independent_event_language_history_action_state_goal_initial_audit_pass=True,costs=dict(task_controls=sum(r['task_controls'] for r in rows),settling_controls=120,policy_queries=sum(r['task_controls'] for r in rows),inference_seconds=sum(r['inference_seconds'] for r in rows),episode_wall_sum_not_end_to_end_seconds=sum(r['wall_seconds'] for r in rows),registered_run_elapsed_seconds=(end-start).total_seconds(),preflight_controls=pre['controlled_steps'],preflight_settling_controls=pre['settling_controls'],preflight_queries=0,preflight_wall_seconds=pre['wall_seconds'],previous_failed_v30_preflight_controls=0,earlier_development_episodes_still_charged=20,earlier_CPU_engineering_queries_still_charged=10,precise_model_load_seconds=None),limits=['All four contexts are exposed development; tiny selected sample, no inference or independent confirmation.','Stage trigger reads privileged goal predicate; home is an engineering servo and low-noise mode changes execution policy. No learned repair or RSI claim.','Original full two-goal success is primary; atomic subgoals are not full-task gains.','Prefix discrepancies must be disclosed and make causal attribution inconclusive; no outcome deletion.','All twelve results, numerical failures and costs retained. If all new arms fail, do not sweep further knobs on this branch.'],new_training=False,work_demonstrated=False)
    folder=W/'analysis';folder.mkdir();(folder/'diagnostic_readout.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),'utf8')
    with (folder/'paired_results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:out[k] for k in ['cells','all_A_prefixes_exact','branch_rejected_if_all_new_arms_fail','costs','work_demonstrated']}))
if __name__=='__main__':main()
