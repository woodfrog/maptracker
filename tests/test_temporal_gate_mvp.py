import unittest

try:
    import torch
    TORCH_OK = True
except Exception:
    TORCH_OK = False

try:
    from plugin.models.transformer_utils.MapTransformer import SlotwiseTemporalGate
    from plugin.models.transformer_utils.MapTransformer import MapTransformerLayer
    GATE_OK = TORCH_OK
except Exception:
    GATE_OK = False

try:
    from plugin.models.mapers.vector_memory import VectorInstanceMemory
    VECTOR_MEMORY_OK = TORCH_OK
except Exception:
    VECTOR_MEMORY_OK = False

try:
    from plugin.models.mapers.MapTracker import MapTracker
    TRACKER_OK = TORCH_OK
except Exception:
    TRACKER_OK = False


class TestTemporalGateMVP(unittest.TestCase):

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_dimension_smoke(self):
        gate = SlotwiseTemporalGate(embed_dims=16, enabled=True)
        q = torch.randn(1, 8, 16)
        mem = torch.randn(3, 8, 16)
        values, alpha = gate(q, mem)
        self.assertEqual(values.shape, mem.shape)
        self.assertEqual(alpha.shape, (1, 8, 3))
        self.assertTrue(torch.isfinite(values).all().item())
        self.assertTrue(torch.isfinite(alpha).all().item())

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_no_history(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=True)
        q = torch.randn(1, 1, 8)
        mem = torch.randn(0, 1, 8)
        values, alpha = gate(q, mem)
        self.assertEqual(values.shape[0], 0)
        self.assertEqual(alpha.shape, (1, 1, 0))

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_one_alpha_parity(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=False)
        q = torch.randn(1, 3, 8)
        mem = torch.randn(4, 3, 8)
        values, alpha = gate(q, mem)
        self.assertTrue(torch.allclose(values, mem))
        self.assertTrue(torch.allclose(alpha, torch.ones_like(alpha)))

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_zero_alpha_suppression(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=True)
        with torch.no_grad():
            gate.gate_mlp[-1].weight.zero_()
            gate.gate_mlp[-1].bias.fill_(-100.0)
        q = torch.randn(1, 4, 8)
        mem = torch.randn(4, 4, 8)
        values, alpha = gate(q, mem)
        self.assertLess(alpha.max().item(), 1e-4)
        self.assertLess(values.abs().max().item(), 1e-4)
        self.assertTrue(torch.isfinite(values).all().item())
        self.assertTrue(torch.isfinite(alpha).all().item())

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_gate_raises_when_q_len_not_one(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=True)
        q = torch.randn(2, 1, 8)
        mem = torch.randn(3, 1, 8)
        with self.assertRaisesRegex(AssertionError, 'q_len == 1'):
            gate(q, mem)

    @unittest.skipUnless(GATE_OK, 'gate/runtime deps unavailable')
    def test_track_idx_boundary_guard_raises(self):
        valid_track_idx = torch.tensor([0, 2], dtype=torch.long)
        with self.assertRaisesRegex(AssertionError, 'tracked-query boundary'):
            MapTransformerLayer._assert_valid_track_idx_in_bounds(valid_track_idx, track_len=2, batch_i=0)

    @unittest.skipUnless(VECTOR_MEMORY_OK, 'vector memory/runtime deps unavailable')
    def test_corrupted_read_isolation(self):
        clean = torch.randn(4, 2, 8)
        canonical = clean.clone()
        key_padding = torch.zeros((2, 4), dtype=torch.bool)
        select_indices = [torch.arange(4), torch.arange(4)]
        corrupt, _, _ = VectorInstanceMemory._build_local_corrupted_read_view(
            clean, canonical, select_indices, key_padding,
            corruption_mode='c_full', stale_offset=2, apply_corruption=True)
        self.assertTrue(torch.allclose(canonical, clean))
        self.assertEqual(corrupt.shape, clean.shape)

    @unittest.skipUnless(VECTOR_MEMORY_OK, 'vector memory/runtime deps unavailable')
    def test_corrupted_read_uses_propagated_selected_source(self):
        clean = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
        canonical = torch.tensor([[[100.0]], [[200.0]], [[300.0]], [[400.0]], [[500.0]]])
        key_padding = torch.zeros((1, 3), dtype=torch.bool)
        select_indices = [torch.tensor([1, 3, 4])]

        corrupt, mask, eligible = VectorInstanceMemory._build_local_corrupted_read_view(
            clean.clone(), canonical, select_indices, key_padding,
            corruption_mode='c_full', stale_offset=2, apply_corruption=True)

        self.assertEqual(corrupt[1, 0, 0].item(), clean[0, 0, 0].item())
        self.assertNotEqual(corrupt[1, 0, 0].item(), canonical[1, 0, 0].item())
        self.assertTrue(mask[0, 1].item())
        self.assertTrue(eligible[0, 1].item())

    @unittest.skipUnless(VECTOR_MEMORY_OK, 'vector memory/runtime deps unavailable')
    def test_corrupted_read_missing_source_is_ineligible(self):
        clean = torch.tensor([[[10.0]], [[20.0]], [[30.0]]])
        canonical = torch.tensor([[[100.0]], [[200.0]], [[300.0]], [[400.0]], [[500.0]]])
        key_padding = torch.zeros((1, 3), dtype=torch.bool)
        select_indices = [torch.tensor([1, 3, 4])]

        corrupt, mask, eligible = VectorInstanceMemory._build_local_corrupted_read_view(
            clean.clone(), canonical, select_indices, key_padding,
            corruption_mode='c_full', stale_offset=1, apply_corruption=True)

        self.assertEqual(corrupt[0, 0, 0].item(), clean[0, 0, 0].item())
        self.assertFalse(mask[0, 0].item())
        self.assertFalse(eligible[0, 0].item())

    @unittest.skipUnless(VECTOR_MEMORY_OK, 'vector memory/runtime deps unavailable')
    def test_c_tail_selectivity(self):
        clean = torch.arange(5 * 1 * 1).view(5, 1, 1).float()
        key_padding = torch.zeros((1, 5), dtype=torch.bool)
        select_indices = [torch.arange(5)]
        corrupt, mask, eligible = VectorInstanceMemory._build_local_corrupted_read_view(
            clean.clone(), clean.clone(), select_indices, key_padding,
            corruption_mode='c_tail', stale_offset=1, tail_keep_recent=2, apply_corruption=True)
        self.assertTrue(mask[0, -1].item() is False)
        self.assertTrue(mask[0, -2].item() is False)
        self.assertTrue((eligible[0, :-2] == mask[0, :-2]).all().item())
        self.assertTrue(torch.allclose(corrupt[-1], clean[-1]))

    @unittest.skipUnless(TRACKER_OK, 'maptracker/runtime deps unavailable')
    def test_inference_track_query_length_from_track_info(self):
        tracker = MapTracker.__new__(MapTracker)
        track_query_info = [
            {'track_query_hs_embeds': torch.randn(3, 8)},
            {'track_query_hs_embeds': torch.randn(0, 8)},
            {},
        ]

        MapTracker._set_inference_tracked_query_length(tracker, track_query_info)

        self.assertEqual(tracker.tracked_query_length, {0: 3, 1: 0, 2: 0})


if __name__ == '__main__':
    unittest.main()
