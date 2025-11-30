import unittest

from torch import nn

from FACTORIZER.model.factorizer.factorizer.factorization.base import MF


class _DummyInit(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.size = size
        self.rank = rank

    def forward(self, x):
        return None


class _DummySolver(nn.Module):
    def __init__(self, size, rank):
        super().__init__()
        self.size = size
        self.rank = rank
        self.flops = 0

    def forward(self, x, factors):
        return factors


class MFUpdateRankTests(unittest.TestCase):
    def test_update_rank_reinitializes_factories(self):
        size = (4, 6)
        initial_rank = 2
        mf = MF(
            size=size,
            rank=initial_rank,
            init=_DummyInit,
            solver=_DummySolver,
            num_iters=1,
            verbose=False,
        )

        previous_init = mf.init
        previous_solver = mf.solver

        new_rank = 4
        mf.update_rank(new_rank)

        self.assertEqual(mf.rank, new_rank)
        self.assertIsNot(previous_init, mf.init)
        self.assertIsNot(previous_solver, mf.solver)
        self.assertEqual(mf.init.rank, new_rank)
        self.assertEqual(mf.solver.rank, new_rank)
        self.assertEqual(mf.init.size, size)
        self.assertEqual(mf.solver.size, size)


if __name__ == "__main__":
    unittest.main()
