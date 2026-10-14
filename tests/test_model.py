"""Unit tests for COBALT."""

import unittest
import torch
from cobalt.model import COBALT

class TestCOBALT(unittest.TestCase):
    def setUp(self):
        self.model = COBALT(spectral_dim=13, foundation_dim=768, hidden_dim=256)
        self.batch_size = 4
        self.seq_len = 100

    def test_forward(self):
        spec = torch.randn(self.batch_size, self.seq_len, 13)
        found = torch.randn(self.batch_size, self.seq_len, 768)
        outputs = self.model(spec, found)
        self.assertIn('logits', outputs)

if __name__ == '__main__':
    unittest.main()
