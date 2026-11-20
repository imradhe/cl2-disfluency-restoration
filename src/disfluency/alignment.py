"""Global edit alignment: inputs are observations, never disfluency labels."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Edit:
    op: str
    expected: int | None
    observed: int | None


def align(expected, observed):
    """Unit-cost Levenshtein alignment; deterministic diagonal-first tie breaking.

    I means an observed phone absent from the transcript's canonical phones;
    D means a canonical phone absent from the observation. Align across word
    boundaries so boundary drift is not automatically charged twice.
    """
    n, m = len(expected), len(observed)
    cost = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i-1][j-1] + (expected[i-1] != observed[j-1]),
                             cost[i-1][j] + 1, cost[i][j-1] + 1)
    result = []
    i, j = n, m
    while i or j:
        if i and j and cost[i][j] == cost[i-1][j-1] + (expected[i-1] != observed[j-1]):
            result.append(Edit("M" if expected[i-1] == observed[j-1] else "S", i-1, j-1))
            i, j = i-1, j-1
        elif j and cost[i][j] == cost[i][j-1] + 1:
            result.append(Edit("I", i, j-1))
            j -= 1
        else:
            result.append(Edit("D", i-1, None))
            i -= 1
    return result[::-1]


def phone_evidence(canonical, observed):
    """Per-word evidence and n+1 boundary gaps from canonical phones per word."""
    flat, owner, first = [], [], {}
    for i, phones in enumerate(canonical):
        if phones:
            first[len(flat)] = i
        flat.extend(phones)
        owner.extend([i] * len(phones))
    words = [dict(realized=[], ops=[], matches=0, substitutions=0,
                  deletions=0, insertions=0) for _ in canonical]
    gaps = [[] for _ in range(len(canonical) + 1)]
    edits = align(flat, observed)
    for edit in edits:
        if edit.op == "I":
            p = edit.expected
            if p == len(flat):
                gaps[-1].append(observed[edit.observed])
            elif p in first:
                gaps[first[p]].append(observed[edit.observed])
            else:
                w = words[owner[p]]
                w['realized'].append(observed[edit.observed])
                w['ops'].append('I')
                w['insertions'] += 1
        else:
            w = words[owner[edit.expected]]
            w['ops'].append(edit.op)
            field = {'M': 'matches', 'S': 'substitutions', 'D': 'deletions'}[edit.op]
            w[field] += 1
            if edit.observed is not None:
                w['realized'].append(observed[edit.observed])
    return {'words': words, 'gaps': gaps, 'distance': sum(e.op != 'M' for e in edits)}
