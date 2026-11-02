"""Prepare/register/run exactly12 post-hoc context diagnostics. No training."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,copy,shutil,os,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1];W=ROOT/'rsi_group_meeting/v35/cross_context_diagnostic'
OLD=ROOT/'rsi_group_meeting/v32/repair_learning_dev';D0=ROOT/'rsi_group_meeting/v30/language_schedule_dev';T=ROOT/'rsi_group_meeting/v31/handoff_factorial_dev'
FIT=ROOT/'rsi_group_meeting/v30/research_agent/offline_teacher_fit_diagnostic/execution'
sys.path.insert(0,str(ROOT/'rsi_group_meeting'))
SENSORS=['agentview_image','robot0_eye_in_hand_image','robot0_joint_pos','robot0_gripper_qpos']
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as stream:json.dump(x,stream,ensure_ascii=False,indent=2)
def rel(p):return Path(p).resolve().relative_to(ROOT).as_posix()
def status(stage,**items):
    temporary=W/'status.working.json'
    temporary.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),stage=stage,expected=12,work_demonstrated=False,**items),ensure_ascii=False,indent=2),'utf8')
    temporary.replace(W/'status.json')
def ready():
    old=read(OLD/'registration.json');analysis=read(OLD/'analysis/repair_readout.json');fit=read(FIT/'summary.json')
    assert read(OLD/'evaluation/collection_complete.json')['physical_episodes']==24 and len(analysis['rows'])==24
    assert analysis['registration_sha256']==sha(OLD/'registration.json')
    assert fit['density_head_forwards_completed']==fit['density_head_forwards_attempted']==9873
    assert fit['frozen_author_parameters_before']==fit['frozen_author_parameters_after']==old['frozen_pretrained_parameters_sha256']
    assert fit['physical_controls']==fit['spatial_encodings']==fit['policy_action_sampling_calls']==fit['backward_or_optimizer_steps']==0
    return old
def register():
    started=time.perf_counter();old=ready();assert not W.exists(),'Preserve existing preparation/attempt; no blind registration retry.'
    d0=read(D0/'registration.json');tp=read(T/'registration.json');trained=read(OLD/'trained_actors.json')
    assert old['parent']['sha256']==d0['parent']['sha256']==tp['parent']['sha256']
    inputs={};sources={};copies=[]
    def bind(path,expected=None,target=None):
        path=Path(path).resolve();value=sha(path)
        if expected is not None:assert value==expected,str(path)
        (sources if target=='source' else inputs)[rel(path)]=value
        return value
    for plan in [old,d0,tp]:
        for name,value in plan['implementation_sha256'].items():bind(ROOT/name,value,'source')
        for asset in plan.get('assets',[]):bind(ROOT/asset['file'],asset['sha256'])
    bind(ROOT/'rsi_group_meeting/v4/xvla/protocol.json')
    for c in old['evaluation_contexts']:
        if c['partition']=='Long_development':
            bind(ROOT/c['setting']['bddl'],c['setting']['bddl_sha256'])
            bind(ROOT/c['setting']['init_file'],c['setting']['init_sha256'])
    for name in ['cross_context_diag_v35.py','cross_context_worker_v35.py','analyze_cross_context_v35.py']:
        bind(ROOT/'rsi_group_meeting'/name,target='source')
    servo_analysis_freeze=read(T/'analysis_frozen_before_development_outcomes.json')
    bind(ROOT/'rsi_group_meeting/analyze_handoff_factorial_v31.py',servo_analysis_freeze['analysis_implementation_sha256'],'source')
    bind(T/'analysis_frozen_before_development_outcomes.json')
    bind(ROOT/'rsi_group_meeting/repair_learning_awake_v32.py',target='source')
    for folder,name in [(OLD,'registration.json'),(OLD,'analysis/repair_readout.json'),(OLD,'evaluation/collection_complete.json'),(OLD,'trained_actors.json'),(OLD,'cache_worker_metadata.json'),(D0,'registration.json'),(T,'registration.json'),(FIT,'summary.json')]:bind(folder/name)
    bind(ROOT/'rsi_group_meeting/v30/research_agent/minimal_cross_diagnostic_interpretation.json')
    actors={'parent':dict(file=old['parent']['file'],sha256=old['parent']['sha256'],all_parameter_digest=read(OLD/'cache_worker_metadata.json')['initial_all_parameter_digest'])}
    for a in ['uniform_bc','local_bc']:
        assert trained[a]['steps']==20
        actors[a]={k:trained[a][k] for k in ['file','sha256','all_parameter_digest']}
    for item in actors.values():bind(item['file'],item['sha256'])
    def reference(folder,key):
        file=folder/'results'/f'{key}.json';r=read(file);files={rel(file):bind(file)}
        for path,h in [(file.with_suffix('.npz'),r['trace_sha256']),(folder/r['events'],r['events_sha256']),(folder/r['video'],r['video_sha256']),(folder/r['initial_sensors'],r['initial_sensors_sha256'])]:files[rel(path)]=bind(path,h)
        return dict(result=rel(file),initial_sensors=rel(folder/r['initial_sensors']),files_sha256=files)
    parent_refs=[];student_cases=[];teacher_cases=[];parity=[];teacher_train_refs=[]
    for e in [row for row in d0['episodes'] if row['arm']=='joint']:
        c=copy.deepcopy(e['context']);c['partition']='original_training_context';ref=reference(D0,e['key']);r=read(ROOT/ref['result']);assert r['actor_sha256']==actors['parent']['sha256'] and r['executed_physics_steps']==900
        first=json.loads((D0/r['events']).read_text('utf8').splitlines()[0]);assert first['language']==c['setting']['language'] and first['history_length']==1
        parent_refs.append(dict(context=c,**ref));parity.append(dict(context=c,initial_source=ref['initial_sensors'],expected_action=first['action'],expected_log_probability=first['log_probability'],action_exact=True,log_probability_abs_tolerance=1e-5))
        for actor in ['uniform_bc','local_bc']:
            case=dict(kind='student_training_context',episode=dict(key=c['context_id']+'__'+actor,context=copy.deepcopy(c),actor=actor),reference=ref)
            student_cases.append(case)
        matching=next(row for row in tp['episodes'] if row['arm']=='home_learned' and row['context']['context_id']==c['context_id'])
        teacher_train_refs.append(dict(context=copy.deepcopy(c),**reference(T,matching['key'])))
    for c0 in old['evaluation_contexts']:
        if c0['partition']!='Long_development':continue
        c=copy.deepcopy(c0);c['policy_seed']=old['evaluation_policy_seed'];c['task_id']=c['setting']['task_id']
        c['subcommands']=copy.deepcopy(next(e['context']['subcommands'] for e in d0['episodes'] if e['context']['task_id']==c['task_id']))
        c['partition']='original_student_test_context';key=c['context_id']+'__home_learned'
        refs={actor:reference(OLD/'evaluation',c0['context_id']+'__'+actor) for actor in ['parent','uniform_bc','local_bc']}
        teacher_cases.append(dict(kind='teacher_test_context',episode=dict(context=c,key=key,arm='home_learned',home=True,low_noise=False),reference=refs['parent'],reused_student_references=refs))
    assert len(student_cases)==8 and len(teacher_cases)==len(parity)==4
    W.mkdir(parents=True)
    def cp(src,dst):
        value=bind(src);dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True);assert not dst.exists();shutil.copy2(src,dst);assert sha(dst)==value
        copies.append(dict(source=rel(src),copy=rel(dst),sha256=value))
    for src in sorted((D0/'canonical').iterdir()):cp(src,W/'student/canonical'/src.name)
    for case in teacher_cases:
        c=case['episode']['context'];s=c['setting'];stem=f"{s['suite']}_task{s['task_id']}_{s['perturbation']}_init{c['state_index']}"
        for suffix in ['.json','.npz']:cp(OLD/'evaluation/canonical'/f'{stem}{suffix}',W/'teacher/canonical'/f'{stem}{suffix}')
    for i,item in enumerate(parity):
        dst=W/'parity_inputs'/f'{i:02d}.npz';cp(ROOT/item['initial_source'],dst);item['observation_file']=str(dst)
    for name in ['trained_actors.json','cache_worker_metadata.json']:cp(OLD/name,W/name)
    p=dict(registered_utc=now(),scope='Post-hoc developmental cross-context diagnosis; training-context student success nevercounts aswork. Original24targetnegativeandlocalsourceharmremainvisible.',
           parent=old['parent'],actors=actors,frozen_pretrained_parameters_sha256=old['frozen_pretrained_parameters_sha256'],public_sensors=SENSORS,
           horizon=900,home_controls=60,teacher_recipe='Unchangedv31.episode:goal0triggersnextactionB; publicpropriohome60 withgain.5; originallearnedNormalsigma; A/Btextunchanged; Bquerieswarmhistory/RNGduringhome.',
           cases=student_cases+teacher_cases,parity=parity,reused_parent_train=parent_refs,reused_teacher_train=teacher_train_refs,
           implementation_sha256=sources,input_files_sha256=inputs,canonical_and_metadata_copies=copies,
           budget=dict(new_physical_episodes=12,maximum_task_controls=10800,settling_controls=120,first_action_parity_queries=4,first_action_parity_spatial_encodings=4,first_action_parity_action_samples=4,parity_physical_controls=0,parity_physical_resets=0,parity_policy_history_rng_resets=4,new_training_steps=0,maximum_feature_packets=10804,additional_model_calls_for_feature_preservation=0),
           feature_packet_rule='Savealreadynativecomputedpre-action4x768F32featureperquery,proposalaction/logp,actualquerylanguage/actor/history/case/step/mode. Nosamplingorqueue/RNGchange. Teacherservopacketsareproposalonly, actualbehaviorlogpnullinevent. query_secondsendbeforepacketIO. PhysicalcasepacketIOisalreadyinepisodewall;paritypacketIOisalreadyinrunwall;neverdoubled.',
           ordering='All8studenttrain-contextcasesinD0registeredorderUthenL,then4teachertest-contextcasesinoriginalv32contextorder. Nooutcome-dependentstoporswap.',
           comparison_limit='Trainingcontextusesinit10/noiseseeds20265300/01;testcontextusesinit11/12/noiseseed20265400. Bothinitialstateandexecutionnoisechange; thisdoesnotisolatepureinitcausality.',
           stopping='All12completeoriginaltasks orregisterednumericfailures. Infrastructurefault:preserveattempt/events/video/partialNPZandexceptionframeprefix; auditbeforeanyrecovery,noblindrestart,noresamplingoroverwriting.',
           original_main_analysis_sha256=sha(OLD/'analysis/repair_readout.json'),offline_fit_summary_sha256=sha(FIT/'summary.json'),
           interpretation_table=read(ROOT/'rsi_group_meeting/v30/research_agent/minimal_cross_diagnostic_interpretation.json'),
           timing='Runelapsedstartsafterverificationandlockacquisition; commandend-to-endwallrequiresexternalobserver. Unknownwholeworkflowpreparationnotzero.',
           registration_cpu_wall_seconds=time.perf_counter()-started,work_demonstrated=False,new_policy_training=False)
    save(W/'registration.json',p);cpreg=W/'teacher/registration.json';shutil.copy2(W/'registration.json',cpreg)
    save(W/'analysis_frozen_before_cross_outcomes.json',dict(frozen_utc=now(),registration_sha256=sha(W/'registration.json'),analyzer_sha256=sha(ROOT/'rsi_group_meeting/analyze_cross_context_v35.py'),new_outcome_files_at_freeze=0))
    print('Prepared registration12newcases+4queryparity; nothinglaunched.',flush=True)
def verify():
    p=read(W/'registration.json')
    for name,value in {**p['implementation_sha256'],**p['input_files_sha256']}.items():assert sha(ROOT/name)==value,name
    for item in p['canonical_and_metadata_copies']:assert sha(ROOT/item['source'])==sha(ROOT/item['copy'])==item['sha256']
    assert sha(W/'registration.json')==sha(W/'teacher/registration.json')
    assert read(W/'analysis_frozen_before_cross_outcomes.json')['analyzer_sha256']==sha(ROOT/'rsi_group_meeting/analyze_cross_context_v35.py')
    return p
def preserve_exception_prefix(error,case):
    tb=error.__traceback__;saved=[]
    while tb:
        values=tb.tb_frame.f_locals
        if tb.tb_frame.f_code.co_name=='episode' and 'actions' in values:
            arrays={};conversion_errors={}
            def keep(name,value):
                try:
                    a=np.asarray(value)
                    if a.dtype==object:raise TypeError('Objectarraywouldrequirepickle; preservingavailability/errorinstead')
                    arrays[name]=a.copy()
                except BaseException as conversion:conversion_errors[name]=repr(conversion)
            for name in ['actions','states','robot','goals','predicates','warnings','successes','proposals','proposal_logps','action_sources','decoder_modes','initialstate','initialgoal','initial_state','initial_pred']:
                if name in values:keep(name,values[name])
            for name in ['action','proposal','sim','state','rs','robotstate']:
                if name in values:keep('latest_local_'+name,values[name])
            for source in ['raw','observed','public']:
                container=values.get(source)
                if isinstance(container,dict):
                    for field in SENSORS+['robot0_eef_pos','robot0_eef_quat']:
                        if field in container:keep('latest_'+source+'_'+field,container[field])
            path=W/'failure_prefixes'/f"{case['episode']['key']}.npz";path.parent.mkdir(parents=True,exist_ok=True)
            assert not path.exists();np.savez_compressed(path,**arrays)
            loop_index=values.get('i',values.get('step'));loop_index=int(loop_index) if isinstance(loop_index,(int,np.integer)) else None
            appended=len(values['actions']);possible=None if loop_index is None else appended<=loop_index
            receipt=dict(file=rel(path),sha256=sha(path),appended_control_count=appended,loop_index_zero_based=loop_index,possible_unappended_final_step=possible,
                         current_iteration_physical_step_completion_unknown=possible is not False,latest_single_step_locals_may_be_stale_or_pre_action=True,
                         failure_source_file=tb.tb_frame.f_code.co_filename,failure_source_line=tb.tb_lineno,
                         array_shapes={k:list(a.shape) for k,a in arrays.items()},array_conversion_errors=conversion_errors,
                         recovery='Onlycapturedexistingtracebacklocals; noenvgetter/encoder/step/resetafterfinallyclose. Do notinfer0executedcontrolsfrommissingappend.')
            save(path.with_suffix('.json'),receipt);saved.append(receipt)
        tb=tb.tb_next
    return saved
def run():
    p=verify();ready();assert not (W/'run_started.json').exists(),'Auditpriorattempt, neverblindlyrestart.'
    from local_run_lock_v4 import exclusive
    from policy_pipe_client_v7 import PolicyPipe
    from repair_learning_awake_v32 import awake
    for name in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[name]='1'
    for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='2'
    class CheckedPipe(PolicyPipe):
        def __init__(self,*args):
            self.query_attempts=0;self.case_key='parity';self.query_phase='parity';self.case_query_counts={};super().__init__(*args)
        def request(self,command,**arguments):
            if command=='query':
                index=self.case_query_counts.get((self.query_phase,self.case_key),0)
                self.case_query_counts[(self.query_phase,self.case_key)]=index+1
                packet=W/'feature_packets'/('parity' if self.query_phase=='parity' else 'cases')/self.case_key/f'{index:04d}.npz'
                assert not packet.exists() and not packet.with_suffix('.json').exists()
                arguments=dict(arguments,feature_packet_file=str(packet),query_case=self.case_key,query_step=index,query_phase=self.query_phase)
                self.query_attempts+=1
                with (W/'head_query_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(phase='attempt',ordinal=self.query_attempts,case=self.case_key,utc=now(),arguments=arguments))+'\n')
            answer=super().request(command,**arguments)
            if command=='query':
                with (W/'head_query_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(phase='completed',ordinal=self.query_attempts,case=self.case_key,utc=now(),answer=answer))+'\n')
            if command in ['load_actor','reset']:
                with (W/'policy_control_receipts.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(utc=now(),command=command,arguments=arguments,answer=answer))+'\n')
            return answer
        def close(self):
            try:
                if self.process.poll() is None:
                    ack=self.request('close');assert ack['closed'] and ack['frozen_sha256']==p['frozen_pretrained_parameters_sha256']
                    assert ack['all_parameter_digest']==p['actors'][ack['actor']]['all_parameter_digest']
                    save(W/'worker_close_receipt.json',ack);self.process.wait(timeout=30)
            finally:self.terminate();self.log.close()
    with exclusive('robotics_gpu'),exclusive('repair_learning_v32'),exclusive('cross_context_v35'),awake():
        start=time.perf_counter();save(W/'run_started.json',dict(started_utc=now(),pid=os.getpid(),registration_sha256=sha(W/'registration.json')))
        status('startup',completed=0,current=None,kind=None,worker_pid=None,parity_completed=0)
        client=None;current=None;done=0
        try:
            stamp=time.perf_counter();client=CheckedPipe(ROOT/'envs/dmpel/Scripts/python.exe',ROOT/'rsi_group_meeting/cross_context_worker_v35.py',W)
            save(W/'worker_metadata.json',dict(metadata=client.metadata,worker_pid=client.worker_pid,model_startup_wall_seconds=time.perf_counter()-stamp))
            status('parity',completed=0,current=None,kind='first_action_query_only',worker_pid=client.worker_pid,parity_completed=0)
            client.request('load_actor',actor='parent');parity=[]
            for i,item in enumerate(p['parity']):
                status('parity',completed=0,current=item['context']['context_id'],kind='first_action_query_only',worker_pid=client.worker_pid,parity_completed=i)
                client.case_key=item['context']['context_id'];client.query_phase='parity'
                save(W/'parity_attempts'/f'{i:02d}.json',dict(started_utc=now(),registered_item=item,physical_controls=0,physical_resets=0))
                client.request('reset',seed=item['context']['policy_seed'])
                q=client.request('query',observation_file=item['observation_file'],language=item['context']['setting']['language'],decode_mode='learned')
                save(W/'parity_receipts'/f'{i:02d}.json',q)
                np.testing.assert_array_equal(np.asarray(q['action'],dtype=np.float32),np.asarray(item['expected_action'],dtype=np.float32))
                assert abs(q['log_probability']-item['expected_log_probability'])<=item['log_probability_abs_tolerance']
                assert q['history_length']==1 and q['actor']=='parent';parity.append(q)
            save(W/'parity_complete.json',dict(completed_utc=now(),queries=4,spatial_encodings=4,action_samples=4,physical_controls=0,physical_resets=0,all_actions_bitwise_equal=True,scope='Onlyfourfirstactionqueries;nofull900stepprefixreplayorphysics.',logp_tolerance=1e-5,inference_seconds=sum(q['query_seconds'] for q in parity)))
            import benchmark_v4
            import evaluate_repair_learning_v32 as student
            import handoff_factorial_dev_v31 as teacher
            import paired_observation_v4 as observation_module
            canonical_original=observation_module.canonicalize
            def checked_canonical(env,raw,setting,state,folder):
                measured,pair=canonical_original(env,raw,setting,state,folder)
                reference=read(ROOT/current['reference']['result'])
                assert pair['model_sha256']==reference['paired_observation']['model_sha256']
                assert pair['full_state_sha256']==reference['paired_observation']['full_state_sha256']
                with np.load(ROOT/current['reference']['initial_sensors']) as expected:
                    for field in SENSORS:np.testing.assert_array_equal(measured[field],expected[field])
                save(W/'initial_input_guards'/f"{current['episode']['key']}.json",dict(checked_utc=now(),case=current['episode']['key'],reference_result_sha256=sha(ROOT/current['reference']['result']),public_arrays_equal=True,model_xml_hash_equal=True,canonical_full_state_hash_equal=True,before_any_case_policy_query_or_task_action=True))
                return measured,pair
            observation_module.canonicalize=checked_canonical
            student.W=W;student.E=W/'student';teacher.WORK=W/'teacher'
            for case in p['cases']:
                current=case;e=case['episode'];save(W/'case_attempts'/f"{e['key']}.json",dict(started_utc=now(),kind=case['kind'],registration_sha256=sha(W/'registration.json')))
                status('collecting',completed=done,current=e['key'],kind=case['kind'],worker_pid=client.worker_pid,parity_completed=4)
                client.case_key=e['key'];client.query_phase='case'
                if case['kind']=='student_training_context':
                    case_p=dict(p);case_p['evaluation_policy_seed']=e['context']['policy_seed'];student.episode(e,case_p,client);folder=W/'student'
                else:
                    client.request('load_actor',actor='parent');teacher.episode(e,p,client);folder=W/'teacher'
                r=read(folder/'results'/f"{e['key']}.json")
                with np.load(folder/r['initial_sensors']) as a,np.load(ROOT/case['reference']['initial_sensors']) as b:
                    for field in SENSORS+['initial_sim_state']:np.testing.assert_array_equal(a[field],b[field])
                    if 'initial_goal_predicates' in b:np.testing.assert_array_equal(a['initial_goal_predicates'],b['initial_goal_predicates'])
                done+=1
                status('collecting',completed=done,current=e['key'],kind=case['kind'],worker_pid=client.worker_pid,parity_completed=4)
            observation_module.canonicalize=canonical_original
            verify();client.close();client=None
            assert (W/'worker_close_receipt.json').exists()
            save(W/'collection_complete.json',dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),physical_episodes=12,settling_controls=120,parity_queries=4,run_elapsed_seconds=time.perf_counter()-start,work_demonstrated=False))
            status('complete_requires_analysis',completed=12,current=None,kind=None,worker_pid=None,parity_completed=4)
        except BaseException as error:
            prefixes=preserve_exception_prefix(error,current) if current else []
            save(W/'failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),completed_episodes=done,current_case=current,preserved_frame_prefixes=prefixes,traceback=traceback.format_exc(),infrastructure_censoring_not_task_failure=True,automatic_retry=False))
            status('stopped_requires_audit',completed=done,current=current['episode']['key'] if current else None,kind=current['kind'] if current else None,worker_pid=client.worker_pid if client else None,parity_completed=len(parity) if 'parity' in locals() else None)
            raise
        finally:
            if 'observation_module' in locals() and 'canonical_original' in locals():observation_module.canonicalize=canonical_original
            if client is not None:client.close()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['register','run']);args=parser.parse_args()
    register() if args.command=='register' else run()
