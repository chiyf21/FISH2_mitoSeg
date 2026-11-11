"""Explain absence per final 3D instance from actual source-track records."""
import numpy as np

def annotate(row,tracks,run,roi=(703,0,8,1215,512,52)):
    for frame in row['frames']:
        z=frame['z'];details=[]
        for tid in row['source_tracks']:
            t=tracks[tid];seed=t['seed']['z'];stop=t.get('stop') or {};reason=stop.get('reason');saved=next((f for f in t['frames'] if f['z']==z),None)
            entry=dict(track_id=tid,seed_z=seed,horizon_last_z=min(seed+10,roi[5]-1),stop=stop)
            bidir=t.get('propagation_mode')=='bidirectional_score_stop'
            if bidir:entry['horizon_last_z']=None
            if not roi[2]<=z<roi[5]:code='outside_roi';label=f'超出本次推理Z范围{roi[2]}–{roi[5]-1}'
            elif z<seed and not bidir:code='before_seed';label=f'尚未开始：seed在Z{seed}，只向后推'
            elif saved:
                with np.load(run/'frames'/f'z{z:03d}_track{tid:05d}.npz') as f:nonempty=bool(f['mask'].any())
                entry['score']=saved['score']
                code='saved_mask' if nonempty else 'empty_mask';label='本层已推理并保存mask' if nonempty else '本层已推理，mask为空'
            elif bidir:
                direction='backward' if z<seed else 'forward';s=t.get('direction_stops',{}).get(direction,{})
                detail=s.get('detail') or s;score=detail.get('score');sz=detail.get('z',s.get('z'))
                if s.get('reason') in ('low_score','refresh_prompt_low_score','crop_replay_low_score'):
                    code='low_score_stop';label=f'{direction}方向在Z{sz}低分停止'
                    if score is not None:entry['rejected_score']=score;label+=f'（拒绝score={score:.4f}）'
                else:code='not_retained';label=f'{direction}方向无保留mask：{s.get("reason","见记录")}'
                entry['direction_stop']=s
            elif reason in ('low_score','refresh_prompt_low_score','crop_replay_low_score'):
                detail=stop.get('detail') or stop;sz=detail.get('z',stop.get('z'));score=detail.get('score')
                if z>seed+10:code='beyond_horizon';label=f'超过该track后推10层上限Z{seed+10}'
                else:
                    code='low_score_stop';label=f'该track在Z{sz}低分停止'
                    if score is not None:label+=f'（拒绝score={score:.4f}）';entry['rejected_score']=score
                    if reason=='crop_replay_low_score':label+='，发生在移窗记忆重放'
                    if reason=='refresh_prompt_low_score':label+='，发生在tight bbox重新提示'
            elif z>seed+10:code='beyond_horizon';label=f'超过该track后推10层上限Z{seed+10}'
            elif reason=='unresolved_crop_margin':code='crop_failure';label='移窗后仍未满足边距，track停止'
            elif reason=='empty_mask_at_refresh':code='refresh_empty';label='重新提示层mask为空，无法取tight bbox，track停止'
            else:code='not_retained';label='本层无保留帧；见该track停止记录'
            details.append(dict(code=code,label=label,**entry))
        frame['track_statuses']=details
        if frame['pred_area']>0:
            label='已推理：本层有该3D实例的mask';code='retained'
        elif not roi[2]<=z<roi[5]:label=f'超出本次推理范围（仅推Z{roi[2]}–{roi[5]-1}）';code='outside_roi'
        elif any(d['code']=='saved_mask' for d in details):label='已推理，但mask归入其他3D结构或被组装过滤；当前实例本层为空';code='assembly_absent'
        elif any(d['code']=='empty_mask' for d in details):label='已推理，模型输出mask为空';code='empty_mask'
        elif any(d['code']=='low_score_stop' for d in details):label='来源track已低分停止，本层无保留mask';code='low_score_stop'
        elif details and all(d['code']=='before_seed' for d in details):label='本实例尚未开始推理：本层早于所有seed';code='before_seed'
        elif any(d['code']=='beyond_horizon' for d in details):label='超过来源track后推10层上限，本层未继续推理';code='beyond_horizon'
        else:label='来源track因移窗或重新提示问题停止，见逐track记录';code='track_stopped'
        frame['inference_status']=code;frame['inference_status_label']=label
