import unittest

import torch
import torch.nn.functional as F

from rvc.vc.rmvpe import RMVPE


class ContextModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_lengths = []

    def forward(self, mel):
        self.input_lengths.append(mel.shape[-1])
        return F.avg_pool1d(mel[:, :1], 65, stride=1, padding=32).transpose(1, 2)


class LongAudioPitchTests(unittest.TestCase):
    def test_long_pitch_inference_preserves_frames_and_context_with_bounded_inputs(self):
        predictor = RMVPE.__new__(RMVPE)
        predictor.device = "cpu"
        predictor.is_half = False
        predictor.model = ContextModel()
        for frames in (3072, 3073, 6307):
            with self.subTest(frames=frames):
                torch.manual_seed(0)
                mel = torch.randn(1, 128, frames)
                expected = predictor._mel2hidden(mel)
                predictor.model.input_lengths.clear()
                actual = predictor.mel2hidden(mel)
                self.assertEqual(actual.shape, (1, frames, 1))
                self.assertLessEqual(max(predictor.model.input_lengths), 3456)
                self.assertTrue(all(length % 32 == 0 for length in predictor.model.input_lengths))
                torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
