# code_base/tests/test_similarity.py
import torch
from utils.similarity import l2_normalize, cosine_sim, cosine_self, cosine_against_ref

def test_l2_normalize_unit_length():
    X = torch.randn(5, 7)
    Xn = l2_normalize(X)
    norms = Xn.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)

def test_cosine_self_identity_diagonal():
    X = torch.randn(4, 8)
    S = cosine_self(X)
    assert torch.allclose(torch.diag(S), torch.ones(4), atol=1e-6)

def test_cosine_sim_symmetry():
    A = torch.randn(3, 6)
    B = torch.randn(5, 6)
    S = cosine_sim(A, B)
    # S_ij should equal cosine(B_j, A_i)
    # Check a random pair
    i, j = 1, 2
    val1 = S[i, j]
    val2 = cosine_sim(B[j:j+1], A[i:i+1]).squeeze()
    assert torch.allclose(val1, val2, atol=1e-6)

def test_against_ref_shape():
    A = torch.randn(10, 16)
    ref = torch.randn(16)
    s = cosine_against_ref(A, ref)
    assert s.shape == (10,)
