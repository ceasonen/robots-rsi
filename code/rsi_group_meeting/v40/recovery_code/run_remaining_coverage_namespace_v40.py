"""Auditable process-only recovery of the one unstarted coverage arm.

Never call the original hardcoded two-arm tail. Scientific preamble, per-arm body,
except and finally remain AST-identical to the frozen original training source.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json,sys,time,traceback

ROOT=Path('F:/robot').resolve()
CODE=ROOT/'rsi_group_meeting/v40/code'
ORIGINAL_W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control'
RECOVERY=ORIGINAL_W/'recovery_namespace_guard_20261102'
WORK=RECOVERY/'remaining_arm'
ADDENDUM=RECOVERY/'recovery_addendum_frozen_before_remaining_arm.json'
INSTALLED_RUNNER=ROOT/'rsi_group_meeting/v40/recovery_code/run_remaining_coverage_namespace_v40.py'
FROZEN_TRAIN=CODE/'train_state_coverage_v40.py'
FROZEN_TRAIN_SHA='5b057f08ee8487f5e7f27becdd7d4222733f88d650c0a3a93e61fd5de675f84e'
ORIGINAL_REG_SHA='e72b0d1d33729ff4c846aa20541efeddbbd2fc78bd9e2b4696b2c4601004940e'
FIRST_CANDIDATE_SHA='953958bb31a0c8b5cde73978426c95ccaa88ec20ac06a40ac867885fc4b33111'
INITIAL_DIGEST='90fd44d0931e906df4c3b16207e118a5678730c273440446855ca433f5c990c9'
COPIED_INPUTS=('registration.json','fixed_training_membership.json','fixed_training_sampling.json','fixed_training_coverage.json')

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def resolved(value):
    p=Path(value);return p.resolve() if p.is_absolute() else (ROOT/p).resolve()
def node_dump(nodes):return ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
def text_sha(text):return hashlib.sha256(text.encode('utf8')).hexdigest()
def main_parts(tree):
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'];assert len(functions)==1
    main=functions[0]
    withs=[n for n in main.body if isinstance(n,ast.With)];assert len(withs)==1
    outer=withs[0];tries=[n for n in outer.body if isinstance(n,ast.Try)];assert len(tries)==1
    tr=tries[0]
    loops=[n for n in tr.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='arm'];assert len(loops)==1
    loop=loops[0];assert isinstance(loop.iter,ast.Name) and loop.iter.id=='ARMS' and not loop.orelse
    return main,outer,tr,loop

def build_remaining_module(source):
    """PureAST transformation, callable byCPU proof without importing the model."""
    assert text_sha(source)==FROZEN_TRAIN_SHA
    original=ast.parse(source,filename=str(FROZEN_TRAIN));tree=copy.deepcopy(original)
    om,ow,ot,ol=main_parts(original);main,outer,tr,loop=main_parts(tree)
    assert isinstance(main.body[0],ast.Assert)
    original_guard=ast.dump(main.body[0],include_attributes=False)
    guard=ast.parse("assert Path(__file__).resolve()==_RECOVERY_INSTALLED_RUNNER, 'Recovery must run from the reviewed installed runner.'").body[0]
    main.body[0]=ast.copy_location(guard,main.body[0])
    i=tr.body.index(loop);prefix=tr.body[:i];removed_tail=tr.body[i+1:]
    assert removed_tail and isinstance(removed_tail[0],ast.Assert)
    # This orchestration assignment invokes the ORIGINAL else-assert in the unchangedbody.
    # Empty originalarms_zero_states stays empty: cross-process torch.equal is NOT claimed.
    seed_identity=ast.parse('initial_digest=_RECOVERY_EXPECTED_INITIAL_DIGEST').body[0]
    seed_identity=ast.copy_location(seed_identity,loop)
    loop.iter=ast.copy_location(ast.Tuple(elts=[ast.Constant(value='coverage_KL')],ctx=ast.Load()),loop.iter)
    finish=ast.parse('_RECOVERY_FINISH(completed_actors,all_parity,initial_digest,began)').body[0]
    finish=ast.copy_location(finish,removed_tail[0])
    tr.body=prefix+[seed_identity,loop,finish]
    # Import/definition prefix is retained; only the old automatic main invocation is removed.
    auto=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
          and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__']
    assert len(auto)==1;tree.body.remove(auto[0]);ast.fix_missing_locations(tree)
    chunks={
      'module_preamble':([n for n in original.body if n is not om and not isinstance(n,ast.If)],
                         [n for n in tree.body if n is not main]),
      'main_scientific_preamble':(om.body[1:om.body.index(ow)],main.body[1:main.body.index(outer)]),
      'with_initialization_preamble':(ow.body[:ow.body.index(ot)],outer.body[:outer.body.index(tr)]),
      'try_scientific_preamble_and_helpers':(ot.body[:ot.body.index(ol)],prefix),
      'per_arm_body':(ol.body,loop.body),
      'exception_handlers':(ot.handlers,tr.handlers),
      'finally':(ot.finalbody,tr.finalbody),
    }
    proof={};dumps={}
    for key,(before,after) in chunks.items():
        left=node_dump(before);right=node_dump(after);assert left==right,key
        proof[key]={'original_sha256':text_sha(left),'recovery_sha256':text_sha(right),'AST_exact':True}
        dumps[key]={'original_ast_dump':left,'recovery_ast_dump':right}
    evidence=dict(frozen_training_source_sha256=FROZEN_TRAIN_SHA,unchanged_chunks=proof,
      per_arm_body_AST_sha256=proof['per_arm_body']['original_sha256'],
      original_tail_AST_sha256=text_sha(node_dump(ot.body[ot.body.index(ol)+1:])),
      recovery_tail_AST_sha256=text_sha(node_dump([finish])),
      original_entry_guard_AST_sha256=text_sha(original_guard),
      recovery_module_AST_sha256=text_sha(ast.dump(tree,include_attributes=False)),
      exact_orchestration_changes=['entrypathguard','globalWremap_aftermoduleimports','prependlocalinitial_digest_fromauditedfirstarm',
        'mainarmloop_iter_coverageonly_globalARMSunchanged','replacehardcodedtwoarmtailwithsinglearmclosure','removeoldautomaticmaincall'],
      new_initial_digest_assignment='initial_digest=_RECOVERY_EXPECTED_INITIAL_DIGEST',
      original_cross_arm_tensor_equal_branch_executed=False,
      replacement_identity_evidence='Originalelseassert matches fullnamedparameterdigest to firstarm; parent/newzerohead8queryparity unchanged.',
      declared_main_arm_schedule=['coverage_KL'],original_global_ARMS_for_preload=['oldA_KL','coverage_KL'])
    compiled=compile(tree,str(FROZEN_TRAIN),'exec') # Compileallnestednonlocalclosures without execution.
    return compiled,evidence,dumps

def validate_before_model(study):
    p=study.verify();assert sha(ORIGINAL_W/'registration.json')==ORIGINAL_REG_SHA and sha(FROZEN_TRAIN)==FROZEN_TRAIN_SHA
    a=read(ADDENDUM)
    assert a['registration_sha256']==ORIGINAL_REG_SHA and a['scientific_changes']==[] and a['remaining_arm']=='coverage_KL'
    assert a['frozen_training_source_sha256']==FROZEN_TRAIN_SHA
    assert a['original_training_failure_sha256']==sha(ORIGINAL_W/'training_failure.json')
    assert a['original_owner_phase_sha256']==sha(ORIGINAL_W/'owner_phase_timings/two_fixed_trainings.json')
    assert a['recovery_sources_sha256'][INSTALLED_RUNNER.name]==sha(INSTALLED_RUNNER)
    for name,h in a['recovery_sources_sha256'].items():
        f=INSTALLED_RUNNER.parent/name if Path(name).name==name else resolved(name)
        assert sha(f)==h,name
    preserved={resolved(path):h for path,h in a['preserved_original_evidence_sha256'].items()}
    for path,h in preserved.items():assert sha(path)==h,str(path)
    assert (ORIGINAL_W/'status.json') not in preserved,'Archiveoldmutablestatusratherthanbindingthelivepath.'
    first=read(ORIGINAL_W/'arms/oldA_KL/training_receipt.json');done=read(ORIGINAL_W/'arms/oldA_KL/training_complete.json')
    assert first['sha256']==FIRST_CANDIDATE_SHA==sha(first['file']) and first['steps']==done['steps']==2000
    assert first['initial_all_parameter_digest']==done['initial_all_parameter_digest']==INITIAL_DIGEST
    assert first['completed_loss_presentations']==64000 and first['completed_distribution_forwards']==128000
    required={f.resolve() for f in (ORIGINAL_W/'arms/oldA_KL').rglob('*') if f.is_file()}
    required|={resolved(first['file']),ORIGINAL_W/'training_failure.json',ORIGINAL_W/'owner_phase_timings/two_fixed_trainings.json'}
    assert required.issubset(preserved),'Addendummustbindallfirstarmartifacts,candidateandoriginalfailure/phase.'
    failure=read(ORIGINAL_W/'training_failure.json');assert failure['arm']=='coverage_KL' and failure['step']==0
    assert failure['completed_arms']==['oldA_KL']
    assert 'native_model_namespaces' in failure['traceback'] and 'name not in sys.modules' in failure['traceback']
    for key in ['loss_presentations_attempted','loss_presentations_completed','distribution_forwards_attempted','distribution_forwards_completed','parity_attempted_current_arm','parity_completed_current_arm']:assert failure[key]==0
    for name in ['trained_actors.json','training_receipt.json','training_complete.json','zero_B_functional_parity_complete.json','evaluation_started.json','evaluation_complete.json','analysis_attempt.json']:assert not(ORIGINAL_W/name).exists(),name
    assert not(ORIGINAL_W/'arms/coverage_KL').exists() and not(ORIGINAL_W/'actors/coverage_KL_step2000.pt').exists()
    assert not list((ORIGINAL_W/'evaluation/results').glob('*.json'))
    return p,a,first

def finish_single_arm(work,study,p,addendum_sha,proof,completed_actors,all_parity,initial_digest,began):
    """Honest one-arm completion only; aggregate creates originalglobalmarkers later."""
    assert set(completed_actors)==set(all_parity)=={'coverage_KL'} and initial_digest==INITIAL_DIGEST
    actor=completed_actors['coverage_KL'];parity=all_parity['coverage_KL'];folder=work/'arms/coverage_KL'
    assert read(folder/'training_receipt.json')==actor and read(folder/'zero_B_functional_parity_complete.json')==parity
    assert actor['initial_all_parameter_digest']==INITIAL_DIGEST and actor['frozen_parameters_sha256']==p['frozen_pretrained_parameters_sha256']
    assert actor['steps']==2000 and actor['completed_loss_presentations']==64000 and actor['completed_distribution_forwards']==128000
    assert actor['teacher_presentations']==actor['retention_state_presentations']==32000 and actor['retention_action_labels_read']==0
    assert Path(actor['file']).resolve().is_relative_to(work/'actors') and sha(actor['file'])==actor['sha256']
    assert parity['total_queries']==parity['total_encoders']==parity['total_action_samples']==8 and parity['physical_controls']==0
    assert actor['sampling_sha256']==p['fixed_training_sampling_sha256']
    study.verify()
    for name in COPIED_INPUTS:assert sha(work/name)==sha(ORIGINAL_W/name)
    info=dict(completed_utc=now(),registration_sha256=ORIGINAL_REG_SHA,recovery_addendum_sha256=addendum_sha,
      arm='coverage_KL',training_runs=1,optimizer_steps=2000,completed_loss_presentations=64000,completed_distribution_forwards=128000,
      teacher_presentations=32000,retention_state_presentations=32000,retention_action_labels_read=0,parity_queries=8,
      actor=actor,actor_file_sha256=actor['sha256'],initial_all_parameter_digest=initial_digest,
      sampling_sha256=p['fixed_training_sampling_sha256'],per_arm_body_AST_sha256=proof['per_arm_body_AST_sha256'],
      original_first_arm_retrained=False,original_first_arm_queries_or_forwards_repeated=0,
      original_cross_process_tensor_equal_executed=False,
      cross_process_initial_full_parameter_digest_checked_by_unchanged_original_else_assert=True,
      within_attempt_elapsed_seconds=time.perf_counter()-began,
      timing_definition='Onlythecurrenttransformedmaininternalattempt; excludesouterrecoveryhash/copy andPythonstartup; notoriginalinterruptedwholetrainingtime.',
      original_global_completion_markers_written=False,whole_project_end_to_end_seconds=None)
    save(work/'one_remaining_arm_receipt.json',info)
    save(work/'one_remaining_arm_complete.json',{k:v for k,v in info.items() if k!='actor'})
    study.status('remaining_coverage_complete_requires_independent_aggregation',completed_arms=['coverage_KL'],original_first_arm_retained=True)

def main():
    assert Path(__file__).resolve()==INSTALLED_RUNNER,'Preparationonly; root mustreview/install/freeze before execution.'
    began=time.perf_counter()
    for path in [ROOT/'scripts',ROOT/'rsi_group_meeting',CODE]:sys.path.insert(0,str(path))
    import state_coverage_study_v40 as study
    p,addendum,first=validate_before_model(study)
    assert not WORK.exists() and not(RECOVERY/'remaining_process_attempt.json').exists(),'No automaticretry.'
    source=FROZEN_TRAIN.read_bytes().decode('utf8');compiled,proof,dumps=build_remaining_module(source)
    assert proof['per_arm_body_AST_sha256']==addendum['per_arm_body_AST_sha256']
    # No sys.modules deletion/rebinding. The originalbackend's namespaceguard remains intact.
    assert not any(n in sys.modules for n in ['libero','libero.lifelong','libero.lifelong.models','libero.lifelong.models.modules']), 'Mustusefreshdedicatedprocess.'
    save(RECOVERY/'remaining_process_attempt.json',dict(started_utc=now(),original_registration_sha256=ORIGINAL_REG_SHA,
      recovery_addendum_sha256=sha(ADDENDUM),runner_sha256=sha(__file__),per_arm_body_AST_sha256=proof['per_arm_body_AST_sha256'],
      new_training_runs_authorized=1,remaining_arm='coverage_KL',automatic_retry=False,first_arm_candidate_sha256=first['sha256']))
    try:
        WORK.mkdir()
        copied=[]
        for name in COPIED_INPUTS:
            raw=(ORIGINAL_W/name).read_bytes()
            with (WORK/name).open('xb') as f:f.write(raw)
            assert sha(WORK/name)==sha(ORIGINAL_W/name);copied.append(dict(file=name,sha256=sha(WORK/name)))
        save(WORK/'copied_frozen_inputs.json',copied);save(WORK/'executed_AST_projection.json',proof)
        namespace={'__name__':'_frozen_train_v40_remaining_process','__file__':str(INSTALLED_RUNNER)}
        exec(compiled,namespace) # Onlyoriginalimports/functiondefinitions, automaticmain removed.
        assert namespace['ARMS']==['oldA_KL','coverage_KL']
        assert namespace['verify'] is study.verify and namespace['status'] is study.status
        namespace['W']=WORK
        namespace['_RECOVERY_INSTALLED_RUNNER']=INSTALLED_RUNNER
        namespace['_RECOVERY_EXPECTED_INITIAL_DIGEST']=INITIAL_DIGEST
        namespace['_RECOVERY_FINISH']=lambda ca,ap,di,be:finish_single_arm(WORK,study,p,sha(ADDENDUM),proof,ca,ap,di,be)
        namespace['main']()
        assert (WORK/'one_remaining_arm_complete.json').exists()
        save(RECOVERY/'remaining_process_command_returned.json',dict(completed_utc=now(),recovery_observed_main_wall_seconds=time.perf_counter()-began,
          definition='Thisrunnerinternalmainintervalcontainsvalidation/copy/transformedmain; excludesPythonstartup/finalreceiptwrites. Externalownercommandwallseparate.',
          honest_remaining_arm_training_runs=1,original_global_training_closed=False,independent_aggregation_required=True))
    except BaseException as error:
        save(RECOVERY/'remaining_process_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),
          observed_main_wall_seconds=time.perf_counter()-began,automatic_retry=False,partialfiles_and_optimizer_tensors_preserved_by_original_exception_logic=True))
        raise

if __name__=='__main__':main()
