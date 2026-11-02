"""Prepare all48 closed v40 recordings for human review, using CPU decode only.

No model, policy query, simulator, replay or rendering is imported. Explicit CLI
opt-in AND the independent root raw-review gate are required before any labels
or recordings are opened. This script does not itself perform human review.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, time, traceback
import numpy as np

ROOT=Path('F:/robot').resolve()
W=ROOT/'rsi_group_meeting/v40/state_coverage_kl_control'
OUT=ROOT/'rsi_group_meeting/v40/root_visual_review'
GATE=ROOT/'rsi_group_meeting/v40/root_completed_review/review.json'
REG_SHA='e72b0d1d33729ff4c846aa20541efeddbbd2fc78bd9e2b4696b2c4601004940e'
NATIVE_SIZE=(256,128)
FRAME_LABELS=('Initial sensor','Predicate0 first true','Predicate0 plus60','By520 or early stop','Predicate0 lost / tail-10','Last recorded action')

class ReviewError(RuntimeError):pass
def require(value,message):
    if not value:raise ReviewError(message)
def read(p):return json.loads(Path(p).read_text('utf8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pixel_sha(rgb):return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
def local_artifact(base,relative):
    p=(Path(base)/relative).resolve()
    require(p.is_relative_to(Path(base).resolve()),'source artifact path stays under evaluation')
    return p
def native_rgb(value):
    value=np.asarray(value)
    require(value.shape==(NATIVE_SIZE[1],NATIVE_SIZE[0],3) and value.dtype==np.uint8,'native256x128 uint8 RGB, no resize or recolor')
    return np.ascontiguousarray(value)

def select_requests(goals,initial_goals):
    """Indices are action steps: initial0, post-action t maps to video index t.

    Include initial predicate truth in first-hit/loss timing; this is a native
    predicate event, not an assertion about skill timing or stable completion.
    """
    g=np.asarray(goals);initial=np.asarray(initial_goals)
    require(g.dtype==np.bool_ and initial.dtype==np.bool_ and initial.ndim==1 and len(initial)>0,'native Boolean predicate vectors')
    require(g.ndim==2 and g.shape[1:]==initial.shape,'trace/initial predicate dimensions')
    n=len(g);sequence=np.r_[initial[0],g[:,0]]
    hits=np.flatnonzero(sequence);first=int(hits[0]) if len(hits) else None
    losses=np.flatnonzero(sequence[:-1] & ~sequence[1:])+1;lost=int(losses[0]) if len(losses) else None
    post_hits=np.flatnonzero(g[:,0]);first_post=int(post_hits[0]+1) if len(post_hits) else None
    second=first if first is not None else min(300,n)
    third=min(first+60,n) if first is not None else min(360,n)
    fifth=lost if lost is not None else max(0,n-10)
    labels=[FRAME_LABELS[0],FRAME_LABELS[1] if first is not None else 'No predicate0; sample300',FRAME_LABELS[2] if first is not None else 'No predicate0; sample360',FRAME_LABELS[3],'Predicate0 first lost' if lost is not None else 'Tail-10 clamped to initial',FRAME_LABELS[5]]
    steps=[0,second,third,min(520,n),fifth,n]
    if n==0:labels=[labels[0]]+['Initial only; zero task actions']*5
    requests=[dict(slot=i,label=label,action_step=int(step),video_frame_index=int(step) if step>0 else None,source_kind='initial_sensors_exact' if step==0 else 'recorded_video_CPU_decoded',native_predicates=(initial if step==0 else g[step-1]).tolist()) for i,(label,step) in enumerate(zip(labels,steps))]
    require(len(requests)==6 and all(0<=q['action_step']<=n for q in requests),'six bounded requests per case')
    return requests,dict(executed_actions=n,first_predicate0_true_including_initial=first,first_post_action_predicate0_true=first_post,first_predicate0_loss_including_initial_transition=lost,unique_source_positions=len(set(steps)),initial_predicates=initial.tolist(),last_post_action_predicates=g[-1].tolist() if n else None,final_observed_predicates=(g[-1] if n else initial).tolist())

def decode_video_cpu(video,n,wanted_steps,allow_zero_action_fallback):
    """Sequentially decode every frame; collector records initial + n actions."""
    import imageio.v2 as imageio
    reader=None;selected={};decoded=0;error=None;began=time.perf_counter()
    try:
        reader=imageio.get_reader(video,format='ffmpeg',input_params=['-hwaccel','none'],output_params=['-threads','2'])
        for index,frame in enumerate(reader):
            frame=native_rgb(frame);decoded+=1
            require(index<=n,'video has extra frames beyond initial+n actions')
            if index in wanted_steps and index>0:selected[index]=frame.copy()
        require(decoded==n+1,'v40 video must contain initial frame plus every executed action')
    except Exception as exc:
        if not (n==0 and allow_zero_action_fallback and decoded<=1):raise
        error=repr(exc)
    finally:
        if reader is not None:reader.close()
    if n>0:require(set(selected)==set(wanted_steps),'all selected post-action frames actually decoded')
    return selected,dict(recorded_video_frames_decoded=decoded,expected_frames_for_v40_initial_plus_actions=n+1,decoder='imageio FFmpeg, sequential raw RGB; -hwaccel none; threads2',complete_stream_decoded=error is None,zero_action_initial_only_fallback=bool(error),zero_action_decode_error=error,CPU_decode_wall_seconds=time.perf_counter()-began)

def save_native_png(path,rgb):
    from PIL import Image
    rgb=native_rgb(rgb);path=Path(path)
    with path.open('xb') as f:Image.fromarray(rgb,'RGB').save(f,format='PNG')
    with Image.open(path) as stored:require(np.array_equal(np.asarray(stored.convert('RGB')),rgb),'PNG preserves source pixels exactly')
    return dict(file=str(path),png_file_sha256=sha(path),raw_RGB_pixel_sha256=pixel_sha(rgb),width=256,height=128)

def build_sheet(items,path,font):
    from PIL import Image,ImageDraw
    require(len(items)==4,'exactly four cases per sheet')
    row_height=208;frame_top=72
    sheet=Image.new('RGB',(1536,row_height*4),'white');draw=ImageDraw.Draw(sheet);cells=[]
    for row,item in enumerate(items):
        top=row*row_height;s900='NA' if item['success900'] is None else str(item['success900'])
        draw.text((3,top+2),f"{item['key']}   by520={item['success520']} / by900={s900}",font=font,fill='black')
        for col,frame in enumerate(item['frames']):
            x=col*256;pred=''.join('1' if b else '0' for b in frame['native_predicates'])
            draw.text((x+2,top+21),frame['label'],font=font,fill='black')
            draw.text((x+2,top+37),f"t={frame['action_step']}  {'initial RGB' if frame['action_step']==0 else 'video index='+str(frame['video_frame_index'])}",font=font,fill='black')
            draw.text((x+2,top+53),'native predicates='+pred,font=font,fill='black')
            with Image.open(frame['file']) as image:
                image=image.convert('RGB');require(image.size==NATIVE_SIZE,'contact sheet uses native dimensions');source=np.asarray(image).copy();sheet.paste(image,(x,top+frame_top))
            require(np.array_equal(np.asarray(sheet.crop((x,top+frame_top,x+256,top+frame_top+128))),source),'sheet crop equals original frame pixels')
            cells.append(dict(case=item['key'],slot=frame['slot'],box=[x,top+frame_top,x+256,top+frame_top+128],raw_RGB_pixel_sha256=pixel_sha(source)))
    with Path(path).open('xb') as f:sheet.save(f,format='PNG')
    return dict(file=str(path),sha256=sha(path),width=1536,height=row_height*4,cases=[v['key'] for v in items],original_frame_pixels_unchanged=True,frame_cells=cells)

def check_gate():
    """This must precede all outcome/recording reads."""
    require(GATE.is_file(),'independent root raw-review gate must exist before extraction')
    gate=read(GATE)
    require(gate['registration_sha256']==REG_SHA==sha(W/'registration.json'),'raw-review registration identity')
    require(len(gate['raw_rows'])==48 and gate['all48_raw_success_initial_three_actor_inputs_event_query_action_hashes_checked'] is True,'all48 independently raw checked')
    require(gate['recovery_owner_four_phases_returned_zero'] is True,'recovery owner four commands closed')
    require(gate['frozen_analysis_sha256']==sha(W/'analysis/coverage_readout.json'),'root gate binds complete frozen analysis')
    require(gate['root_visual_review_pending'] is True,'this extraction is preparation for pending human review')
    return gate

def main():
    began=time.perf_counter();process_began=time.process_time();gate=check_gate()
    require(not OUT.exists(),'never overwrite an earlier visual-review extraction')
    protocol=read(W/'registration.json');analysis=read(W/'analysis/coverage_readout.json');complete=read(W/'evaluation_complete.json')
    require(len(protocol['cases'])==len(analysis['rows'])==complete['physical_episodes']==48,'complete48 before media processing')
    require(analysis['registration_sha256']==complete['registration_sha256']==REG_SHA,'complete/analysis registration identity')
    require(read(W/'evaluation_worker_close.json')['closed'] is True,'final worker close ACK')
    require(len({e['key'] for e in protocol['cases']})==48 and all('actor' in e and 'context' in e and 'episode' not in e for e in protocol['cases']),'registration has48 DIRECT episodes, not v38 nested cases')
    require({row['key'] for row in analysis['rows']}=={e['key'] for e in protocol['cases']},'all cases in frozen analysis')
    analysis_by_key={row['key']:row for row in analysis['rows']}
    OUT.mkdir();save(OUT/'extraction_attempt.json',dict(started_utc=now(),implementation_sha256=sha(__file__),registration_sha256=REG_SHA,root_raw_gate_sha256=sha(GATE),automatic_retry=False,case_rule='All48 registration cases in original order, six deterministic slots each. No success-based case filtering.'))
    manifest=[];sheets=[];base=W/'evaluation'
    try:
        for e in protocol['cases']:
            require(Path(e['key']).name==e['key'],'safe case filename')
            result=base/'results'/f"{e['key']}.json";r=read(result);trace=result.with_suffix('.npz')
            require(sha(result)==analysis_by_key[e['key']]['result_sha256'],'source result anchored to frozen analysis')
            video=local_artifact(base,r['video']);initial=local_artifact(base,r['initial_sensors']);events=local_artifact(base,r['events'])
            require(r['episode']==e and r['registration_sha256']==REG_SHA,'exact original registered case identity')
            for path,h in [(trace,r['trace_sha256']),(video,r['video_sha256']),(initial,r['initial_sensors_sha256']),(events,r['events_sha256'])]:require(sha(path)==h,'closed source artifact SHA')
            with np.load(trace,allow_pickle=False) as z:
                g=z['diagnostic_subgoals'].copy();init_g=z['initial_goal_predicates'].copy();n=len(g)
                require(len(z['actions'])==len(z['official_success'])==n==r['executed_physics_steps'],'trace action count')
            requests,diagnostic=select_requests(g,init_g)
            with np.load(initial,allow_pickle=False) as z:
                a=z['agentview_image'];b=z['robot0_eye_in_hand_image']
                require(a.shape==b.shape==(128,128,3) and a.dtype==b.dtype==np.uint8,'two saved native camera arrays')
                initial_rgb=native_rgb(np.concatenate((a[::-1],b[::-1]),axis=1))
                require(np.array_equal(z['initial_goal_predicates'],init_g),'saved initial predicates match trace')
            numerical=bool(r['policy_numerical_failure'])
            require(n>0 or numerical,'zero-action completion must be retained numerical failure, not fabricated task outcome')
            wanted={q['action_step'] for q in requests if q['action_step']>0}
            decoded,video_info=decode_video_cpu(video,n,wanted,allow_zero_action_fallback=numerical)
            frames=[]
            for q in requests:
                rgb=initial_rgb if q['action_step']==0 else decoded[q['action_step']]
                file=OUT/f"{e['key']}__frame{q['slot']+1}.png"
                frames.append(dict(**q,**save_native_png(file,rgb)))
            require(sha(video)==r['video_sha256'],'source video unchanged during full decode')
            manifest.append(dict(key=e['key'],actor=e['actor'],partition=e['context']['partition'],source_result_file=str(result),source_result_sha256=sha(result),source_trace_sha256=sha(trace),source_video_file=str(video),source_video_sha256=sha(video),source_video_bytes=video.stat().st_size,source_initial_sensors_sha256=sha(initial),source_events_sha256=sha(events),success520=r['success_by520'],success900=r['success_by900'],policy_numerical_failure=numerical,**diagnostic,**video_info,frames=frames))
        from PIL import ImageFont
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
        for block in range(12):sheets.append(build_sheet(manifest[block*4:(block+1)*4],OUT/f'all48_block{block+1:02d}.png',font))
        require(len(manifest)==48 and len(sheets)==12 and sum(len(v['frames']) for v in manifest)==288,'all48/12sheets/288slots')
        out=dict(extracted_utc=now(),registration_sha256=REG_SHA,analysis_sha256=sha(W/'analysis/coverage_readout.json'),root_raw_gate_file=str(GATE),root_raw_gate_sha256=sha(GATE),implementation_sha256=sha(__file__),scope='All48 closed recordings decoded on CPU, no outcome-based filtering; six deterministic slots per case. Original initial RGB uses saved sensors; MP4 frames are lossily recorded pixels, not claimed equal to original sensors. No extra model/encoder/simulator/replay/rendering.',frame_mapping='v40 collector records initial video index0 then action t post-state at video index t; n+1 frames. Sensor-source t0 is saved exact separately and not asserted pixel-equal to compressed video0.',frame_rule='Initial; first predicate0 true including initial, else min300,n; min(first+60,n), else min360,n; min520,n; first true->false including initial transition, else max0,n-10; final n. Native predicate order is not skill order. Repeated positions are explicit, not independent samples.',cases=manifest,sheets=sheets,decoded_video_files=48,selected_slots=288,unique_source_positions_sum=sum(v['unique_source_positions'] for v in manifest),wall_seconds_of_CPU_decode_sum=sum(v['CPU_decode_wall_seconds'] for v in manifest),CPU_only_pipeline_wall_seconds=time.perf_counter()-began,python_process_CPU_seconds_excludes_ffmpeg_children=time.process_time()-process_began,ffmpeg_children_CPU_seconds=None,nested_timer_warning='Decode wall is contained in total pipeline wall; do not add them. Python process CPU excludes FFmpeg child CPU.',new_policy_queries=0,new_training_encoder_calls=0,new_simulator_controls=0,new_GPU_renders=0,visual_review_completed=False,human_frames_viewed_by_script=0)
        save(OUT/'manifest.json',out)
        print('All48 recordings prepared:288 deterministic slots,12 sheets. Human visual inspection remains pending.')
    except BaseException as error:
        save(OUT/'extraction_failure_requires_audit.json',dict(failed_utc=now(),error=repr(error),traceback=traceback.format_exc(),completed_cases=len(manifest),completed_sheets=len(sheets),CPU_only_pipeline_wall_seconds=time.perf_counter()-began,automatic_retry=False,preserve_all_partial_outputs=True,visual_review_completed=False))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-after-root-raw-review',action='store_true',help='Explicit execution opt-in, still requires the full root raw-review gate.')
    args=parser.parse_args()
    require(args.run_after_root_raw_review,'Preparation only by default; no source outcomes or media opened.')
    main()
