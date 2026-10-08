import torch
from crf_beam import beam_search
from collections import namedtuple


def max_grad(x, dim=0):
    return torch.zeros_like(x).scatter_(dim, x.argmax(dim, True), 1.0)


semiring = namedtuple('semiring', ('zero', 'one', 'mul', 'sum', 'dsum'))
Log = semiring(zero=-1e38, one=0., mul=torch.add, sum=torch.logsumexp, dsum=torch.softmax)
Max = semiring(zero=-1e38, one=0., mul=torch.add, sum=(lambda x, dim=0: torch.max(x, dim=dim)[0]), dsum=max_grad)


def scan(Ms, idx, v0, S:semiring=Log):
    T, N, C, NZ = Ms.shape
    alpha = Ms.new_full((T + 1, N, C), S.zero)
    alpha[0] = v0
    for t in range(T):
        alpha[t+1] = S.sum(S.mul(Ms[t], alpha[t, :, idx]), dim=-1)
    return alpha


class CTC_CRF_Torch:

    def __init__(self, state_len=5, alphabet='NACGT'):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def forward_scores(self, scores, S: semiring=Log):
        T, N, C = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        v0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return scan(Ms, self.idx.to(torch.int64), v0, S)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        vT = scores.new_full((N, self.n_base**(self.state_len)), S.one)
        idx_T = self.idx.flatten().argsort().reshape(*self.idx.shape)
        Ms_T = scores[:, :, idx_T]
        idx_T = torch.div(idx_T, self.n_base + 1, rounding_mode='floor')
        return scan(Ms_T.flip(0), idx_T.to(torch.int64), vT, S).flip(0)
    
    def posteriors(self, scores, S: semiring=Log):
        fwd = model.forward_scores(scores, S)
        bwd = model.backward_scores(scores, S)
        return torch.softmax(fwd + bwd, dim=-1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        paths = traceback.argmax(2) % len(self.alphabet)
        return paths


def compute_scores_torch(scores):
    model = CTC_CRF_Torch()
    fwd = model.forward_scores(scores)
    bwd = model.backward_scores(scores)
    posts = torch.softmax(fwd + bwd, dim=-1)

    return {
        'scores': scores.transpose(0, 1).cpu(),
        'bwd': bwd.transpose(0, 1).cpu(),
        'posts': posts.transpose(0, 1).cpu(),
    }


def decode_cpu(x, beam_width=32, beam_cut=100.0, scale=1.0, offset=0.0, blank_score=2.0, rna=False):
    sequence, qstring, moves = beam_search(x['scores'], x['bwd'], x['posts'], beam_size=beam_width)
    return {
        'sequence': sequence,
        'qstring': qstring,
        'moves': moves,
    }
