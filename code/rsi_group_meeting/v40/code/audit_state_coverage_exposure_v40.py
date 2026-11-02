"""Bounded metadata-only exposure refresh before first new target reset/query."""
from pathlib import Path
import os,re,json,time
from state_coverage_study_v40 import ROOT,W,read,sha,save,now

SKIP={'loss','metrics','reward','rewards','success','successes','success_by520','success_by900','success_by400','official_success','diagnostic_subgoals','goal_predicates','action','actions','scores','score','predictions','advantages','implementation_sha256','input_files_sha256','sources_sha256','receipt_hashes','state_vectors','model_state'}
PATTERN=re.compile(r'(?P<suite>libero_90|libero_10|Long)_task(?P<task>\d+)_(?P<pert>original|swap|task)_init(?P<state>\d+)_noise(?P<seed>\d+)')

def contexts(obj,aliases,suite=None,pert=None):
    """Yield only contextual identifiers; never return outcome/action values."""
    if isinstance(obj,dict):
        s=obj.get('setting',{});s=s if isinstance(s,dict) else {}
        su=obj.get('suite',s.get('suite',suite));pe=obj.get('perturbation',s.get('perturbation',pert))
        task=obj.get('task_id',s.get('task_id'));initial=obj.get('state_index',obj.get('init_index'))
        guessed=set()
        for v in [obj.get('task_name'),obj.get('language'),s.get('task_name'),s.get('language')]:
            if isinstance(v,str):guessed.update(aliases.get(v,set()))
        if isinstance(task,int):guessed={v for v in guessed if v[1]==task}
        if su is not None:guessed={v for v in guessed if v[0]==su}
        if pe is not None:guessed={v for v in guessed if v[2]==pe}
        if len(guessed)==1:
            a,b,c=next(iter(guessed));su=a if su is None else su;task=b if task is None else task;pe=c if pe is None else pe
        if su in ['libero_90','libero_10'] and isinstance(task,int) and isinstance(initial,int) and pe in ['original','swap','task']:yield (su,task,pe,initial)
        for k,v in obj.items():
            if k in SKIP:continue
            if isinstance(v,str) and k in ['key','context_id','context','case_key','registered_context_id']:
                for m in PATTERN.finditer(v):yield ('libero_10' if m['suite']=='Long' else m['suite'],int(m['task']),m['pert'],int(m['state']))
            elif isinstance(v,(dict,list)):yield from contexts(v,aliases,su,pe)
    elif isinstance(obj,list):
        for v in obj:
            if isinstance(v,(dict,list)):yield from contexts(v,aliases,suite,pert)

def run_audit(p):
    began=time.perf_counter();folder=W/'pre_evaluation_exposure_refresh'
    assert not folder.exists(),'Audit a previous refresh before any recovery.'
    save(folder/'attempt.json',dict(started_utc=now(),registration_sha256=sha(W/'registration.json'),scope='Metadata refresh before targetreset; trainingparity already used oldcontexts only. No model/sim/query/action.'))
    baseline=read(W/'frozen_proposals/exposure_sources.json')
    targets={(c['setting']['suite'],c['setting']['task_id'],c['setting']['perturbation'],c['state_index']) for c in p['conditions']};assert len(targets)==16
    aliases={}
    for c in p['conditions']:
        s=c['setting'];identity=(s['suite'],s['task_id'],s['perturbation'])
        for name in [s['task_name'],s['language']]:aliases.setdefault(name,set()).add(identity)
    baseline_by_path={Path(x['path']).resolve():x for x in baseline};files=set(baseline_by_path)
    prune={'feature_packets','training_metrics','training_sampling_steps','parity_records','parity_inputs','__pycache__','videos','figures','renders','node_modules'}
    for parent,dirs,names in os.walk(ROOT/'rsi_group_meeting'):
        path=Path(parent).resolve()
        dirs[:]=[d for d in dirs if d not in prune and not (path/d).resolve().is_relative_to(W)]
        if path.is_relative_to(W):continue
        for name in names:
            if not name.endswith('.json'):continue
            f=path/name
            if ('results' in f.parts or 'attempts' in f.parts or 'probe_receipt' in name or name.startswith('scores_before_any') or name.startswith('committed_decisions_before') or PATTERN.search(f.stem)):
                files.add(f)
    matches=[];changed=[];errors=[];rows=[]
    for file in sorted(files):
        try:
            value=sha(file);old=baseline_by_path.get(file)
            if old is not None and value!=old['sha256']:changed.append(dict(file=str(file),old_sha256=old['sha256'],now_sha256=value))
            kind=old['class'] if old else ('attempt_only_reset_or_action_unconfirmed' if 'attempts' in file.parts else 'completed_or_probe_metadata')
            # Baseline planned/asset metadata is rehashed but never treated as actual execution.
            inspect=old is None or old['class'] in ['completed_trajectory_metadata','attempt_only_reset_or_actions_unconfirmed','probe_or_choice_metadata']
            if inspect:
                found=set(contexts(read(file),aliases))
                for m in PATTERN.finditer(file.stem):found.add(('libero_10' if m['suite']=='Long' else m['suite'],int(m['task']),m['pert'],int(m['state'])))
                for context in sorted(found & targets):matches.append(dict(file=str(file),sha256=value,context=context,classification=kind))
            rows.append(dict(file=str(file),sha256=value,classification=kind,baseline=old is not None))
        except BaseException as err:errors.append(dict(file=str(file),error=repr(err)))
    save(folder/'source_metadata_hashes.json',rows)
    receipt=dict(completed_utc=now(),registration_sha256=sha(W/'registration.json'),target_conditions=16,files_checked=len(rows),new_metadata_files=sum(not x['baseline'] for x in rows),source_metadata_sha256=sha(folder/'source_metadata_hashes.json'),exact_actual_or_attempt_matches=matches,changed_baseline_files=changed,read_or_parse_errors=errors,new_model_or_encoder_or_gradient_or_reset_or_action_calls=0,outcome_fields_selected=False,observed_cpu_io_seconds=time.perf_counter()-began,limits='Bounded local logs only. No guarantee about pretraining, external/unlogged exposure, or concurrent unlogged activity. Tasks/assets were alreadyknown.')
    save(folder/'receipt.json',receipt)
    assert not errors and not changed and not matches,'Exposure refresh conflict; preserve evidence, never replace conditions or blindretry.'
    return receipt
