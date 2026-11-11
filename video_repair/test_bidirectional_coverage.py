import numpy as np
from sequential_rules import mask_box_overlap,box_foreground_fraction
m=np.ones((10,10),bool)
a=mask_box_overlap(m,[0,0],[0,0,5,10]);assert a['iom']==.5 and a['opm']==1 and a['overlap_max']>.5
a=mask_box_overlap(m,[0,0],[0,0,20,20]);assert a['iom']==1 and a['opm']==.25
a=mask_box_overlap(m,[0,0],[5,0,15,10]);assert a['overlap_max']==.5 and not a['overlap_max']>.5
a=mask_box_overlap(m,[100,200],[100,200,105,210]);assert a['opm']==1
a=mask_box_overlap(np.zeros((1,1),bool),[0,0],[0,0,10,10]);assert a['overlap_max']==0
a=mask_box_overlap(m,[0,0],[-10,-10,10,10]);assert a['box_area']==400 and a['opm']==.25
assert box_foreground_fraction(m,[-10,-10,10,10])==.25
n=m.copy();n[:4]=False;assert box_foreground_fraction(n,[0,0,10,10])==.6
assert box_foreground_fraction(n,[0,4,10,10])==1
print('PASS: directional denominators, strict >.5, native origins, full box area and empty masks')
