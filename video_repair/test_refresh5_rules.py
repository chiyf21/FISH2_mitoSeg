import numpy as np
from refresh5_rules import tight_prompt,assemble26
m=np.zeros((20,30),bool);m[4:9,7:13]=1
assert tight_prompt(m,[100,200])==[107,204,113,209]
assert tight_prompt(np.zeros((2,2),bool),[0,0]) is None
v=np.zeros((4,70,70),bool);v[0:2,0:25,0:25]=1;v[0,40:70,0:40]=1
v[2:4,60:62,60:62]=1
labels,keep,count=assemble26(v);assert len(keep)==1 and keep[0][2]==1250 and count==3
# Diagonal inter-slice contact joins; a detached same-plane island does not.
w=np.zeros((2,5,5),bool);w[0,0,0]=1;w[1,1,1]=1;w[0,4,4]=1
l,k,n=assemble26(w,min_voxels=1);assert n==2 and len(k)==1 and k[0][2]==2
print('PASS exact tight bbox/no padding; 26CC, diagonal joins, isolated single-slice/volume filtering')
