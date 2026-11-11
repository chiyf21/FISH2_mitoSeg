import unittest
import numpy as np
from stitch_core import starts,mask_iou,assemble


def frame(z,x,score,mask):
    return {'z':z,'origin_xy':[x,0],'mask':np.array(mask,bool),'score':score,'detection_score':.9}


class StitchTests(unittest.TestCase):
    def test_end_coverage_and_small_block(self):
        self.assertEqual(starts(11,6,4),[0,4,5]);self.assertEqual(starts(3,6,4),[0])
    def test_global_coordinates_and_full_union(self):
        a=frame(0,0,.3,[[1,1,1,1]]);b=frame(0,2,.8,[[1,1,1,1]])
        self.assertAlmostEqual(mask_iou(a,b),2/6)
        self.assertEqual(mask_iou(a,frame(0,10,.8,[[1]])),0)
    def test_shared_z_and_score_winner(self):
        tracks=[{'frames':[frame(z,0,s,[[1,1]]) for z in [0,1]]} for s in [.3,.9]]
        objects,audit=assemble(tracks)
        self.assertEqual(len(objects),1);self.assertEqual(len(audit['duplicate_edges']),2)
        self.assertTrue(all(f['score']==.9 for frames in objects.values() for f in frames))
    def test_no_match_across_disjoint_z(self):
        objects,_=assemble([{'frames':[frame(0,0,.5,[[1]])]},{'frames':[frame(1,0,.9,[[1]])]}])
        self.assertEqual(len(objects),2)
    def test_threshold_is_strict(self):
        objects,audit=assemble([{'frames':[frame(0,0,.5,[[1,1,1,1]])]},{'frames':[frame(0,0,.9,[[1,1,1,1,1]])]}])
        self.assertEqual(len(objects),2);self.assertEqual(len(audit['duplicate_edges']),0)
    def test_transitive_groups_preserve_direct_edge_audit(self):
        tracks=[{'frames':[frame(z,x,s,[[1]*10]) for z in (0,1)]} for x,s in ((0,.1),(1,.9),(2,.3))]
        objects,audit=assemble(tracks)
        self.assertEqual(len(objects),1);self.assertEqual(len(audit['duplicate_edges']),4)
        self.assertTrue(all(f['score']==.9 for v in objects.values() for f in v))


if __name__=='__main__':unittest.main()
