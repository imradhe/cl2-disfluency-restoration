from model.infer import decode_units, unit_prf


def test_decode_units_bio():
    tags = ["B-REPETITION", "I-REPETITION", "B-FILLER_HESITATION", "O", "B-PARTIAL_WORD"]
    assert decode_units(tags) == [(0, 2, "REPETITION"), (2, 3, "FILLER_HESITATION"), (4, 5, "PARTIAL_WORD")]


def test_decode_units_type_change_without_b_starts_new_unit():
    assert decode_units(["B-FILLER_DM", "I-FILLER_DM", "I-REPAIR"]) == [(0, 2, "FILLER_DM"), (2, 3, "REPAIR")]


def test_decode_units_leading_i_and_all_o():
    assert decode_units(["I-FALSE_START", "I-FALSE_START"]) == [(0, 2, "FALSE_START")]
    assert decode_units(["O", "O"]) == []


def gold(*units):
    return [list(units)]


def pred(*units):
    return [[{"s": s, "e": e, "type": t, "restoration": r} for s, e, t, r in units]]


def test_unit_prf_strict_and_lenient():
    g = gold((0, 2, "REPETITION", "COPY_R1"), (2, 3, "FILLER_HESITATION", "FIL:uh"))
    p = pred((0, 3, "REPETITION", "COPY_R1"), (2, 3, "FILLER_HESITATION", "FIL:um"))
    r = unit_prf(g, p, overlap=0.5)
    rep, fil = r["per_type"]["REPETITION"], r["per_type"]["FILLER_HESITATION"]
    assert rep["strict"]["f1"] == 0.0 and rep["lenient"]["f1"] == 1.0     # span too long: IoU 2/3
    assert fil["strict"]["f1"] == 1.0
    assert r["strict_matches"] == 1 and r["restoration_acc_on_strict_matches"] == 0.0   # uh != um


def test_unit_prf_ignores_fluent_O_units_and_counts_false_positives():
    g = gold((0, 3, "O", "SELF"))
    p = pred((0, 3, "PROLONGATION", "SELF"))
    r = unit_prf(g, p)
    assert r["per_type"]["PROLONGATION"]["predicted"] == 1
    assert r["per_type"]["PROLONGATION"]["support"] == 0
    assert r["macro_f1_strict"] == 0.0
