import unittest
import numpy as np
from utils.person_noise import replace_rectangle, stabilize_rectangles, replace_yuv_rectangle


class NoiseInputTest(unittest.TestCase):
    def test_noise_independent_of_original_person_and_background_preserved(self):
        a = np.full((30,40,3),81,dtype=np.uint8)
        b = a.copy();b[5:25,8:32] = [10,200,30]
        box = [8,5,32,25]
        out_a = replace_rectangle(a,box,0,0,57,0)
        out_b = replace_rectangle(b,box,0,0,57,0)
        np.testing.assert_array_equal(out_a,out_b)
        np.testing.assert_array_equal(out_a[:5],a[:5])
        np.testing.assert_array_equal(out_a[25:],a[25:])
        np.testing.assert_array_equal(out_a[:, :8],a[:, :8])
        np.testing.assert_array_equal(out_a[:, 32:],a[:, 32:])
        patch = out_a[5:25,8:32]
        self.assertEqual(set(np.unique(patch)),{0,255})
        np.testing.assert_array_equal(patch[:,:,0],patch[:,:,1])
        np.testing.assert_array_equal(patch[:,:,0],patch[:,:,2])

    def test_deterministic_but_changes_between_frames(self):
        source = np.zeros((30,40,3),dtype=np.uint8)
        a = replace_rectangle(source,[8,5,32,25],0,0,67,0)
        np.testing.assert_array_equal(a,replace_rectangle(source,[8,5,32,25],0,0,67,0))
        self.assertFalse(np.array_equal(a,replace_rectangle(source,[8,5,32,25],0,0,67,1)))

    def test_stabilization_reduces_jitter_without_losing_coverage(self):
        boxes = np.array([[40+i*2+(-1)**i*3,30,80+i*2+(-1)**i*3,110] for i in range(29)])
        result,expanded = stabilize_rectangles(boxes,(240,160))
        self.assertTrue(np.all(result[:,:2]<=expanded[:,:2]))
        self.assertTrue(np.all(result[:,2:]>=expanded[:,2:]))
        self.assertTrue(np.all(result%2==0))
        old_center=(boxes[:,:2]+boxes[:,2:])/2
        new_center=(result[:,:2]+result[:,2:])/2
        self.assertLess(np.abs(np.diff(new_center,n=2,axis=0)).mean(),
                        np.abs(np.diff(old_center,n=2,axis=0)).mean())

    def test_single_frame_and_boundary_clipping(self):
        result,expanded=stabilize_rectangles([[1,2,29,27]],(32,32))
        self.assertTrue(np.all(result>=0) and np.all(result<=32))
        self.assertTrue(np.all(result[:,:2]<=expanded[:,:2]))
        self.assertTrue(np.all(result[:,2:]>=expanded[:,2:]))

    def test_yuv_background_planes_unchanged_outside_aligned_rectangle(self):
        width,height=16,12
        source=np.arange(width*height*3//2,dtype=np.uint8)
        result=np.frombuffer(replace_yuv_rectangle(source.tobytes(),(width,height),[4,2,10,8],0,0,57,0),np.uint8)
        changed=np.zeros_like(source,dtype=bool)
        y=changed[:width*height].reshape(height,width);y[2:8,4:10]=True
        u=changed[width*height:width*height*5//4].reshape(height//2,width//2);u[1:4,2:5]=True
        v=changed[width*height*5//4:].reshape(height//2,width//2);v[1:4,2:5]=True
        np.testing.assert_array_equal(result[~changed],source[~changed])
        patch=result[:width*height].reshape(height,width)[2:8,4:10]
        self.assertEqual(set(np.unique(patch)),{16,235})

    def test_fixed_scene_noise_stable_in_overlapping_moving_rectangles(self):
        width,height=40,30
        raw=np.zeros(width*height*3//2,dtype=np.uint8).tobytes()
        a=np.frombuffer(replace_yuv_rectangle(raw,(width,height),[4,4,24,24],0,0,57,0,
            fixed_scene_noise=True),np.uint8)[:width*height].reshape(height,width)
        b=np.frombuffer(replace_yuv_rectangle(raw,(width,height),[8,6,28,26],0,0,57,1,
            fixed_scene_noise=True),np.uint8)[:width*height].reshape(height,width)
        np.testing.assert_array_equal(a[6:24,8:24],b[6:24,8:24])


if __name__ == '__main__':
    unittest.main()
