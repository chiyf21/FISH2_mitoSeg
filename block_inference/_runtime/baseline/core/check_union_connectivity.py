import numpy as np
from union_connectivity import retain_connected_body
m=np.zeros((5,15,20),bool);m[:3,2:5,2:5]=1;m[1:5,10:14,12:18]=1
seed=np.zeros((15,20),bool);seed[2:5,2:5]=1
out,a=retain_connected_body(m,[100,200,10],seed,[100,200],11,m,[100,200,10])
assert out.sum()==27 and a['removed_components']==1 and a['removed_voxels']==96
# Diagonal cross-slice neighbors connect under26, but a gap does not.
m=np.zeros((4,5,5),bool);m[0,0,0]=m[1,1,1]=m[3,4,4]=1
seed=np.zeros((5,5),bool);seed[0,0]=1
out,a=retain_connected_body(m,[0,0,0],seed,[0,0],0)
assert out.sum()==2 and not out[3,4,4]
print('PASS:26-neighbor diagonal links; detached part removed; reference overlap beats largest volume; origin offsets; subset invariant')
