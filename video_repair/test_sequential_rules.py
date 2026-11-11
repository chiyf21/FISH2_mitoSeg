import numpy as np
from sequential_rules import mask_box_iom,filter_boxes
m=np.ones((10,10),bool);boxes=[dict(global_detection_id=0,box_xyxy=[0,0,8,10]),dict(global_detection_id=1,box_xyxy=[0,0,9,10]),dict(global_detection_id=2,box_xyxy=[-100,-100,100,100])]
assert mask_box_iom(m,[0,0],boxes[0]['box_xyxy'])==.8
r=filter_boxes(boxes,[(7,m)],[0,0]);assert not r[0]['suppressed'] and r[1]['suppressed'] and r[2]['suppressed'];assert r[2]['iom']==1.
assert not any(r['suppressed'] for r in filter_boxes(boxes,[],[0,0]))
assert mask_box_iom(np.zeros((3,3),bool),[0,0],[0,0,3,3])==0
assert mask_box_iom(m,[100,200],[100,200,109,210])==.9
print('PASS:mask denominator, strict>0.8, empty/first-frame all seeds, native origins, frozen propagated-mask snapshot')
