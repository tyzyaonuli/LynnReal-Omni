import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'script'))
from eval_vae_reconstruction import padded_length,geometry,Writer,ssim_rgb


class ReconstructionTests(unittest.TestCase):
    def test_temporal_tail_and_native_decoder_geometry(self):
        for n in range(1,20000):
            p=padded_length(n)
            self.assertEqual(p%17,0)
            self.assertGreaterEqual(p-12,n)
            self.assertEqual(((p//17*5-3)//5)*17+5,p-12)
            if p>34: self.assertLess(p-17-12,n)

    def test_no_upscale_and_even_pixels(self):
        self.assertEqual(geometry(1920,1080,768),(1366,768))
        self.assertEqual(geometry(1280,720,768),(1280,720))
        self.assertEqual(geometry(1920,1080,0),(1920,1080))

    def test_variable_timestamps_preserved(self):
        import av
        import numpy as np
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.mp4'
            c={'fps':'30000/1001','width':64,'height':64,'time_base':'1/90000','pts':[9000,12003,18009,21012]}
            w=Writer(p,c);w.write(np.zeros((4,64,64,3),dtype=np.uint8));w.close()
            with av.open(str(p)) as v:
                frames=list(v.decode(video=0))
                self.assertEqual(len(frames),4)
                for f,pts in zip(frames,c['pts']):
                    self.assertAlmostEqual(float(f.pts*f.time_base),(pts-9000)/90000,places=5)

    def test_ssim_identity_and_changed_image(self):
        import numpy as np
        import torch
        pixels=np.full((1,16,16,3),128,dtype=np.uint8)
        x=torch.from_numpy(pixels).float()/255
        self.assertAlmostEqual(ssim_rgb(x,pixels,'cpu')[0],1,places=5)
        self.assertLess(ssim_rgb(x*.5,pixels,'cpu')[0],.9)


if __name__=='__main__': unittest.main()
