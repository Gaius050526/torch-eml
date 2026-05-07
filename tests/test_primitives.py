"""Tests for verified EML primitive constructions."""

import math
import torch
import pytest

from torch_eml.primitives import (
    EMLExp, EMLLn, EMLSin, EMLCos, EMLPi,
    verify_constructions,
)


class TestEMLExp:
    def test_matches_torch_exp(self):
        x = torch.tensor([0.0, 0.5, 1.0, 2.0, -1.0])
        result = EMLExp()(x)
        expected = torch.exp(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_exp_zero_is_one(self):
        x = torch.tensor([0.0])
        assert abs(EMLExp()(x).item() - 1.0) < 1e-6


class TestEMLLn:
    def test_matches_torch_log(self):
        z = torch.tensor([0.5, 1.0, 2.0, math.e, 10.0])
        result = EMLLn()(z)
        expected = torch.log(z)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_ln_one_is_zero(self):
        z = torch.tensor([1.0])
        assert abs(EMLLn()(z).item()) < 1e-5

    def test_ln_e_is_one(self):
        z = torch.tensor([math.e])
        assert abs(EMLLn()(z).item() - 1.0) < 1e-5


class TestEMLSin:
    def test_matches_torch_sin(self):
        x = torch.tensor([0.0, 0.5, 1.0, math.pi / 2, math.pi])
        result = EMLSin()(x)
        expected = torch.sin(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_sin_zero(self):
        x = torch.tensor([0.0])
        assert abs(EMLSin()(x).item()) < 1e-6

    def test_sin_pi_half(self):
        x = torch.tensor([math.pi / 2])
        assert abs(EMLSin()(x).item() - 1.0) < 1e-6


class TestEMLCos:
    def test_matches_torch_cos(self):
        x = torch.tensor([0.0, 0.5, 1.0, math.pi / 2, math.pi])
        result = EMLCos()(x)
        expected = torch.cos(x)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_cos_zero(self):
        x = torch.tensor([0.0])
        assert abs(EMLCos()(x).item() - 1.0) < 1e-6


class TestEMLPi:
    def test_returns_pi(self):
        x = torch.tensor([0.0])
        result = EMLPi()(x)
        assert abs(result.item() - math.pi) < 1e-5

    def test_broadcasts(self):
        x = torch.randn(5)
        result = EMLPi()(x)
        assert result.shape == x.shape
        assert torch.allclose(result, torch.full_like(x, math.pi), atol=1e-5)


class TestVerifyAll:
    def test_all_constructions_pass(self):
        assert verify_constructions(verbose=False)
