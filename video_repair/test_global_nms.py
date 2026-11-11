from sequential_rules import global_slice_nms

def box(i,b,score,z=0):return dict(global_detection_id=i,box_xyxy=b,slice_score=score,z=z)
a=box(1,[0,0,10,10],.9);b=box(2,[1,0,11,10],.8);c=box(3,[0,0,10,10],.7,1)
kept,audit=global_slice_nms([b,c,a]);assert [r['global_detection_id'] for r in kept]==[1,3]
assert next(r for r in audit if r['detection_id']==2)['nms_suppressor']==1
# IoU exactly .6 survives: intersection75 / union125.
kept,_=global_slice_nms([a,box(4,[2.5,0,12.5,10],.8)]);assert len(kept)==2
# Stable ties, original unmodified boxes, and no cross-slice suppression.
d=box(0,[0,0,10,10],.9);kept,_=global_slice_nms([a,d,c]);assert kept[0] is d
# A suppresses B but not C; B must not transitively suppress C.
kept,_=global_slice_nms([a,b,box(5,[3,0,13,10],.7)]);assert len(kept)==2
print('PASS global same-Z NMS: strict threshold, stable scores, unchanged coordinates, no transitive merge')
