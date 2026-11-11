from detection_graph import group_detections,box_metrics
def d(z,b):return dict(z=z,box_xyxy=b,slice_score=.9)
assert box_metrics([0,0,6,10],[2,0,8,10])[0]==.5
assert len(group_detections([d(0,[0,0,6,10]),d(1,[2,0,8,10])])[0])==2
assert box_metrics([0,0,5,10],[1,-5,11,15])[1]==.8
assert len(group_detections([d(0,[0,0,5,10]),d(1,[1,-5,11,15])])[0])==2
g,e=group_detections([d(0,[0,0,10,10]),d(1,[2,2,6,6]),d(2,[2,2,5,5])]);assert len(g)==1 and g[0]['seed_id']==0
assert len(group_detections([d(0,[0,0,10,10]),d(2,[0,0,10,10])])[0])==2
assert len(group_detections([d(0,[0,0,10,10]),d(0,[0,0,10,10])])[0])==2
g,e=group_detections([d(0,[0,0,10,10]),d(1,[2,0,12,10]),d(2,[4,0,14,10]),d(0,[20,20,30,30])]);assert len(g)==2 and g[0]['member_count']==3
print('PASS:strictIoU/IOM boundaries; smaller-box denominator; transitive/global merge; adjacentZ only; largest-area prompt; no same-plane extra merge')
