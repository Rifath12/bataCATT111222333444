"""Check the optional close-microphone export and block boundaries."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from close_refinement import BLEND, load_parameters, predict, refine
import close_waveform as waveform


class LearnedRefinementTest(unittest.TestCase):
    def setUp(self):
        self.parameters = load_parameters()

    def test_blend_retains_expected_weights_and_shape(self):
        audio = np.random.default_rng(11).normal(size=(3,3000)).astype(np.float32)*.03
        baseline = audio * .8
        result = refine(audio,baseline,self.parameters)
        learned = predict(audio,self.parameters)
        np.testing.assert_array_equal(result,baseline + BLEND*(learned-baseline))
        self.assertEqual(result.shape,audio.shape)

    def test_short_signal_preserves_baseline(self):
        audio = np.zeros((3,320),dtype=np.float32)
        baseline = audio.copy()
        np.testing.assert_array_equal(refine(audio,baseline,self.parameters),baseline)

    def test_chunked_export_matches_continuous(self):
        rng = np.random.default_rng(260)
        audio = rng.normal(size=(3,39_843)).astype(np.float32)*.03
        audio[0] += audio[1]*.15
        audio[2] += audio[1]*.18
        matrix = np.tile(np.eye(3,dtype=np.float32),(waveform.FFT//2+1,1,1))
        matrix[:,0,1] = .04
        matrix[:,2,1] = .06
        original = waveform.suppress(audio,matrix,waveform.FROZEN_PROFILE)
        expected = refine(audio,original,self.parameters)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root/f"input_{label}.wav" for label in "ABC"]
            for filename, values in zip(paths,audio,strict=True):
                sf.write(filename,values,16000,subtype="FLOAT")
            with patch.object(waveform,"REFINED_CORE_SAMPLES",8192):
                waveform.render(paths,root/"out",matrix,
                                refinement=partial(refine,parameters=self.parameters))
            for label, reference in zip("ABC",expected,strict=True):
                written, rate = sf.read(root/"out"/f"speaker_{label}.wav",dtype="float32")
                self.assertEqual(rate,16000)
                self.assertEqual(written.shape,reference.shape)
                np.testing.assert_allclose(written,reference,rtol=5e-5,atol=6e-7)

if __name__ == "__main__":
    unittest.main()
