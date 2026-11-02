"""Synthetic indexing/pixel tests only; no real video/result/analysis opened."""
from pathlib import Path
import importlib.util,json,hashlib,inspect,time
import numpy as np
from PIL import ImageFont
from imageio.plugins.ffmpeg import FfmpegFormat
D=Path(__file__).parent;file=D/'extract_closed_frames_cpu.py';sp=importlib.util.spec_from_file_location('frame_prepare',file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
checks=[]
def check(name,test):assert test;checks.append(dict(name=name,passed=True))
def selected(g,init):return m.select_requests(np.asarray(g,dtype=bool),np.asarray(init,dtype=bool))
r,d=selected([[False,False]]*600,[False,False]);check('no_hit_fixed300_360_and520_tail590_last600',[v['action_step'] for v in r]==[0,300,360,520,590,600] and d['first_predicate0_true_including_initial'] is None)
r,d=selected([[False],[True],[True],[False]]+[[False]]*96,[False]);check('postaction_first_true2_maps_video_index2_and_loss4',[v['action_step'] for v in r]==[0,2,62,100,4,100] and r[1]['video_frame_index']==2)
r,d=selected([[False],[False],[True]],[True]);check('initial_true_is_step0_and_initial_to_first_loss_is_step1',[v['action_step'] for v in r]==[0,0,3,3,1,3] and d['first_predicate0_true_including_initial']==0 and d['first_post_action_predicate0_true']==3)
r,d=selected([[False]],[False]);check('short_rollout_clamps_to_initial_without_negative_video_index',[v['action_step'] for v in r]==[0,1,1,1,0,1])
r,d=m.select_requests(np.zeros((0,2),dtype=bool),np.array([False,True],dtype=bool));check('zero_action_six_initial_only_slots_no_fabricated_post_action',all(v['action_step']==0 and v['source_kind']=='initial_sensors_exact' and v['video_frame_index'] is None for v in r) and d['last_post_action_predicates'] is None)
try:m.select_requests(np.zeros((2,1),dtype=float),np.array([False]))
except m.ReviewError:checks.append(dict(name='non_boolean_predicate_trace_rejected',passed=True))
else:raise AssertionError('nonBoolean accepted')
params=inspect.signature(FfmpegFormat.Reader._open).parameters;check('installed_reader_accepts_explicit_CPU_input_output_params','input_params' in params and 'output_params' in params)
TMP=D/f'synthetic_pixel_test_{time.time_ns()}';TMP.mkdir();items=[]
for i in range(4):
 frames=[]
 for j in range(6):
  rgb=np.zeros((128,256,3),dtype=np.uint8);rgb[:,:,0]=i*40;rgb[:,:,1]=j*30;rgb[:,:,2]=np.arange(256,dtype=np.uint8)[None,:]
  q=dict(slot=j,label='Synthetic predicate test',action_step=j,video_frame_index=j if j else None,native_predicates=[True,False])
  frames.append(dict(**q,**m.save_native_png(TMP/f'case{i}_slot{j}.png',rgb)))
 items.append(dict(key=f'SYNTHETIC_NOT_ROBOT_CASE_{i}',success520=False,success900=None,frames=frames))
sheet=m.build_sheet(items,TMP/'synthetic_sheet.png',ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12));check('24_native_frame_cells_preserve_exact_RGB_under_external_captions',len(sheet['frame_cells'])==24 and sheet['original_frame_pixels_unchanged'] and sheet['width']==1536 and sheet['height']==832)
try:m.save_native_png(TMP/'case0_slot0.png',np.zeros((128,256,3),dtype=np.uint8))
except FileExistsError:checks.append(dict(name='existing_frame_never_overwritten',passed=True))
else:raise AssertionError('overwrite accepted')
out=dict(scope='Synthetic indices/native-pixel composition and installed decoder API signature only; no real MP4 decode, result label or root analysis/gate opened.',source_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),passed=len(checks),failed=0,checks=checks,actual_extraction_executed=False,real_recordings_decoded=0,models_encoders_simulations_or_GPU_renders=0,synthetic_outputs=str(TMP))
(D/'frame_preparation_cpu_receipt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n','utf8');print(json.dumps(dict(passed=len(checks),failed=0,source_sha256=out['source_sha256'],actual_extraction_executed=False)))
