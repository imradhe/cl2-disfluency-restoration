#!/usr/bin/env python3
"""Audit the local MS98 archive and existing project labels without extracting it.

Run from any directory: .venv/bin/python v2/switchboard_stats.py
Only Python's standard library and the project's token parser are required.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from swb.markers import parse_token  # noqa: E402

OUT = Path(__file__).resolve().parent / "reports"
ARCHIVE = ROOT / "data/raw/switchboard_ms98/switchboard_word_alignments.tar.gz"
LABELLED = ROOT / "data/processed/swb_tokens.jsonl.gz"
FILE_RE = re.compile(r"sw(\d+)([AB])-ms98-a-(word|trans)\.text$")
HESITATIONS = "uh um ah hm huh eh er hmm mm umm uhh ehh ugh ooh ohh ho uh-oh uhh-oh tsk".split()
BACKCHANNELS = "uh-huh um-hum huh-uh hum-um".split()


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(rows):
    """Compare retained raw tokens and times with the cached labelled data."""
    h = hashlib.sha256()
    for raw, start, end in rows:
        h.update(json.dumps([raw, round(start, 3), round(end, 3)]).encode())
        h.update(b"\n")
    return h.hexdigest()


def write_csv(name, header, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    kinds, forms, subtypes, cuts, vocabulary = (Counter() for _ in range(5))
    durations = Counter()
    sides = defaultdict(dict)
    raw_fingerprints = {}
    utterances = set()
    malformed = negative = zero = 0
    print("Scanning raw archive...", flush=True)
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            match = FILE_RE.fullmatch(Path(member.name).name)
            if not match or not member.isfile():
                continue
            conv, channel, level = match.groups()
            sid = f"sw{conv}{channel}"
            info = sides[sid]
            info.update(conv=conv, side=channel)
            info[level + "_files"] = info.get(level + "_files", 0) + 1
            rows = []
            for line in tar.extractfile(member).read().decode("latin-1").splitlines():
                parts = line.split(None, 3)
                if not parts:
                    continue
                if len(parts) < (4 if level == "word" else 3):
                    malformed += 1
                    continue
                uid, start, end = parts[:3]
                start, end = float(start), float(end)
                info[level + "_rows"] = info.get(level + "_rows", 0) + 1
                if level == "trans":
                    utterances.add(uid)
                    continue
                token = parse_token(parts[3])
                dur = end - start
                negative += dur < 0
                zero += dur == 0
                info["start"] = min(info.get("start", start), start)
                info["end"] = max(info.get("end", end), end)
                kinds[token.kind] += 1
                durations[token.kind] += dur
                if token.is_speech:
                    forms[token.spoken] += 1
                    vocabulary[token.spoken] += 1
                if token.noise:
                    subtypes[f"{token.kind}:{token.noise}"] += 1
                if token.cut:
                    cuts[f"{token.kind}:{token.cut}"] += 1
                if token.kind not in {"SILENCE", "ASIDE", "UNCLEAR"}:
                    rows.append((token.raw, start, end))
            if level == "word":
                assert sid not in raw_fingerprints, f"Duplicate side: {sid}"
                raw_fingerprints[sid] = fingerprint(rows)

    assert all(s["word_files"] == s["trans_files"] == 1 for s in sides.values())
    conversations = defaultdict(list)
    for s in sides.values():
        conversations[s["conv"]].append(s)
    assert all({s["side"] for s in ss} == {"A", "B"} for ss in conversations.values())
    assert malformed == negative == 0

    print("Auditing existing labelled tokens against raw tokens...", flush=True)
    labels, labelled_kinds, by_label, side_labels = Counter(), Counter(), defaultdict(Counter), Counter()
    labelled_sides = set()
    with gzip.open(LABELLED, "rt", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            sid, tokens = record["sid"], record["tokens"]
            assert sid not in labelled_sides
            labelled_sides.add(sid)
            assert fingerprint((t["raw"], t["start"], t["end"]) for t in tokens) == raw_fingerprints[sid], sid
            present = set()
            for t in tokens:
                label = t["label"]
                labels[label] += 1
                labelled_kinds[t["kind"]] += 1
                by_label[label][t["spoken"] or t["raw"].lower()] += 1
                present.add(label)
            side_labels.update(present)
    assert labelled_sides == set(sides)
    assert labelled_kinds == Counter({k: v for k, v in kinds.items() if k not in {"SILENCE", "ASIDE", "UNCLEAR"}})
    previous = json.loads((ROOT / "data/processed/label_stats.json").read_text())
    assert dict(labels) == previous["labels"]

    n = sum(labels.values())
    speech = sum(v for k, v in kinds.items() if k not in {"NOISE", "SILENCE", "ASIDE", "UNCLEAR"})
    conv_hours = sum(max(s["end"] for s in ss) - min(s["start"] for s in ss) for ss in conversations.values()) / 3600
    side_hours = sum(s["end"] - s["start"] for s in sides.values()) / 3600
    summary = {
        "release": "ISIP/MS-State manually corrected Switchboard word alignments (local OpenSLR SLR5 archive)",
        "source_files": {str(p.relative_to(ROOT)): sha256(p) for p in [ARCHIVE, LABELLED, ROOT / "configs/labels.yaml", ROOT / "src/swb/markers.py", ROOT / "src/swb/label.py"]},
        "conversations": len(conversations), "speaker_sides": len(sides),
        "utterance_rows": sum(s["trans_rows"] for s in sides.values()), "unique_utterance_ids": len(utterances),
        "word_alignment_rows": sum(kinds.values()), "speech_tokens": speech,
        "normalized_speech_vocabulary": len(vocabulary), "retained_project_tokens": n,
        "conversation_annotation_span_hours": conv_hours, "speaker_side_annotation_span_hours": side_hours,
        "summed_speech_alignment_hours": sum(v for k, v in durations.items() if k not in {"NOISE", "SILENCE", "ASIDE", "UNCLEAR"}) / 3600,
        "raw_kinds": dict(kinds), "raw_noise_subtypes": dict(subtypes), "raw_partial_subtypes": dict(cuts),
        "lexical_hesitation_candidates": {w: forms[w] for w in HESITATIONS},
        "lexical_backchannel_forms": {w: forms[w] for w in BACKCHANNELS},
        "project_primary_label_tokens": dict(labels),
        "sides_with_project_label": dict(side_labels),
        "quality_checks": {"malformed_rows": malformed, "negative_duration_word_rows": negative, "zero_duration_word_rows": zero, "raw_and_cached_tokens_match_all_sides": True, "cached_label_counts_match_previous_summary": True},
    }
    (OUT / "switchboard_stats.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv("disfluency_counts.csv", ["project_label", "token_count", "percent_retained_tokens", "speaker_sides"],
              ((k, v, round(100*v/n, 6), side_labels[k]) for k, v in labels.most_common()))
    write_csv("label_token_forms.csv", ["project_label", "normalized_token", "token_count"],
              ((label, word, count) for label in sorted(by_label) if label != "O" for word, count in by_label[label].most_common()))
    write_csv("raw_annotation_counts.csv", ["category", "subtype", "token_count"],
              [("kind", k, v) for k, v in kinds.most_common()] + [("noise", k, v) for k, v in subtypes.most_common()] + [("partial", k, v) for k, v in cuts.most_common()])
    write_csv("lexical_fillers.csv", ["category", "normalized_token", "token_count"],
              [("hesitation_candidate", w, forms[w]) for w in HESITATIONS] + [("backchannel_form", w, forms[w]) for w in BACKCHANNELS] + [("ambiguous_interjection_not_labelled_as_hesitation", "oh", forms["oh"])])

    text = ["# Switchboard corpus audit — v2", "", "## Scope and counting", "",
            "Computed from the local ISIP/MS-State transcript archive and independently recounted from the existing project-labelled token file. No audio is included. This is this release's coverage, not a claim about every Switchboard release. Speaker sides are conversation channels, not unique people; unique speakers cannot be established from these files alone.", "",
            "Occurrences below are **token counts**, not disfluency episode counts. A three-word repeated reparandum contributes three labelled tokens. Structural episode boundaries are not supplied by this release or by the cached primary labels. Each retained token has one primary project label, so overlapping phenomena can be suppressed by label precedence.", "",
            "## Corpus statistics", "", "| Measure | Count |", "|---|---:|"]
    measures = [("Conversations", len(conversations)), ("Speaker sides", len(sides)),
                ("Utterance rows", summary["utterance_rows"]), ("Unique utterance IDs", len(utterances)),
                ("Word-alignment rows (including silence/noise)", sum(kinds.values())),
                ("Speech tokens (including partial/laughing/mispronounced words)", speech),
                ("Normalized speech vocabulary", len(vocabulary)), ("Retained project tokens (speech + noise)", n)]
    text += [f"| {k} | {v:,} |" for k, v in measures]
    text += [f"| Conversation annotation spans | {conv_hours:,.3f} hours |", f"| Speaker-side annotation spans | {side_hours:,.3f} hours |", f"| Summed speech-token alignment durations | {summary['summed_speech_alignment_hours']:,.3f} hours |", "",
             "Conversation span = latest aligned end minus earliest aligned start across both channels, summed once per conversation. Side span uses each channel separately. These include internal silences and are transcript-derived coverage estimates, not measured recording lengths. Speech-duration sums can include overlapping speech and alignment artifacts.", "",
             "## Native transcript annotations", "", "| Parsed annotation | Tokens |", "|---|---:|"]
    text += [f"| {k} | {v:,} |" for k, v in kinds.most_common()]
    text += ["", "Noise and partial-word subtypes (subsets of the counts above):", "", "| Subtype | Tokens |", "|---|---:|"]
    text += [f"| {k} | {v:,} |" for k, v in list(subtypes.most_common()) + list(cuts.most_common())]
    text += ["", "`PARTIAL` is a marked word fragment. `MISPRON` is the project's name for `[spoken/intended]` substitutions; the spelling annotation alone does not establish the acoustic cause. Some such tokens also contain truncation. `NOISE_WORD` is speech through laughter/noise. Noise, laughter, silence, coinages, and pronunciation variants are contextual annotations, not automatically disfluencies.", "",
             "## Lexical hesitation and backchannel forms", "", "These are normalized lexical occurrences, not a contextual judgement that every occurrence is disfluent. Backchannels are often appropriate interactional responses. Zero means no matching spelling in this release; it does not establish absence in the audio. `oh` is shown separately because it is ambiguous and the existing labeler does not tag it as a hesitation.", "", "| Category | Form | Tokens |", "|---|---|---:|"]
    text += [f"| Hesitation candidate | {w} | {forms[w]:,} |" for w in HESITATIONS]
    text += [f"| Backchannel form | {w} | {forms[w]:,} |" for w in BACKCHANNELS]
    text += [f"| Ambiguous interjection | oh | {forms['oh']:,} |", "", "## Existing project labels", "",
             "These counts audit the existing v1 labels as a starting point for v2; they are **not newly established gold annotations**. Raw tokens and rounded timestamps were matched on every side before counting. Label correctness has not been manually validated here. A match to raw tokens verifies coverage, not the accuracy of the inferred label.", "",
             "| Type | Labelled tokens | % of retained tokens | Basis |", "|---|---:|---:|---|"]
    basis = {"O": "Unlabelled by the rules; not verified fluent", "FILLER_HESITATION": "Lexical inventory; contextual ambiguity remains", "FILLER_BACKCHANNEL": "Lexical forms + short-utterance heuristic", "FILLER_DM": "Context and pause heuristics", "REPETITION": "Sequence-matching heuristic", "REPAIR": "Near-copy/interruption heuristic", "FALSE_START": "Truncation/restart heuristic", "PARTIAL_WORD": "Explicit fragments, including truncated substitutions", "PROLONGATION": "Word-duration outlier heuristic", "MISPRONUNCIATION": "Explicit substitutions without truncation", "NOISE": "Explicit nonspeech markers; not a disfluency type"}
    text += [f"| {k} | {v:,} | {100*v/n:.3f}% | {basis[k]} |" for k, v in labels.most_common()]
    text += ["", "Repetitions, repairs, false starts, and prolongations have **no native gold count available in this release**. The table supplies the existing project's inferred token counts. Stuttering blocks, acoustically confirmed sound prolongations, individual click/tsk and ugh types, and other untranscribed events cannot be exhaustively counted from this transcript release; unavailable counts must not be reported as zero. Explicit silence intervals are not automatically hesitation pauses.", "",
             "## Reproduction and checks", "", "Run `.venv/bin/python v2/switchboard_stats.py` from the project root. The archive is streamed without extraction. Existing v1 files are read only.", "",
             f"Checked {len(sides):,} complete A/B sides, identical retained raw tokens/times between archive and cache, and exact agreement with the earlier label summary. Malformed rows: {malformed}; negative-duration word rows: {negative}; zero-duration word rows: {zero}.", "",
             "Machine-readable outputs: `switchboard_stats.json` (including source SHA-256 hashes), `disfluency_counts.csv`, `raw_annotation_counts.csv`, `lexical_fillers.csv`, and `label_token_forms.csv` (every normalized token form per non-O project label). The source paths and rule/config hashes describe the current workspace; the cached labels' original generation version is not established by those hashes.", ""]
    (OUT / "switchboard_stats.md").write_text("\n".join(text))
    print(json.dumps({k: v for k, v in summary.items() if k not in {"source_files"}}, indent=2))


if __name__ == "__main__":
    main()
