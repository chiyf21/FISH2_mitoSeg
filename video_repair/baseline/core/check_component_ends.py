import numpy as np
from repair_component_ends import plan,tight_prompt,union_volumes
m=np.zeros((3,80,140),bool)
m[0,2:22,2:27]=1 # Exactly500 must not trigger.
m[0,40:61,50:75]=1 #525 triggers independently.
m[1,5:20,5:20]=1
m[2,2:17,2:22]=1;m[2,40:55,70:90]=1 #Two300 components: sum600 must not trigger.
specs,audit=plan(m,[100,200,10],250)
caps=[s for s in specs if s['role']!='maxclean']
assert len(caps)==1 and caps[0]['directions']==['backward'] and caps[0]['prompt']['seed_mask_area']==525
assert caps[0]['prompt']['box_xyxy']==[150,240,175,261] and caps[0]['prompt']['padding_native_pixels']==0
assert not any(a['eligible'] for a in audit if a['side']=='tail')
assert not any(s['role'].startswith('head') for s in plan(m,[100,200,0],250)[0])
patch=np.ones((1,2,2),bool);u,o=union_volumes([(m,[100,200,10]),(patch,[98,198,8])]);assert o==[98,198,8]
assert np.array_equal(u[2:5,2:82,2:142],m) and u[0,:2,:2].all()
print('PASS: component area >500 (not sum;500 excluded), boundary excluded, exact tight bbox NO padding, outward direction, original union')
