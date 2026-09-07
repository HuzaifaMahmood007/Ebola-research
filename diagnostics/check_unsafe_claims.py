"""check_unsafe_claims.py -- run the audit's "unsafe to claim" list over the manuscript.

`Reports/Phase0_to_Now_Audit.md` lists sentences that must never appear in the paper. Each one is a
claim the artifacts do not support. This is the mechanical half of that check: every item that can
be detected from the text is a rule below, with the reason it is unsafe and what to say instead.

Items that need a human read are listed at the end as MANUAL, not silently dropped. A checker that
quietly skips the hard half is worse than none, because it reads as a clean pass.

Each rule is (id, why it is unsafe, regex, allow-regex). A hit on `pat` is a finding UNLESS the same
line also matches `ok`, which is how a sentence that names the unsafe claim in order to REFUTE it
passes. That exemption is the main source of false negatives, so `--strict` reports those too.

    conda run -n ebola-train python -m diagnostics.check_unsafe_claims [--strict]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC = ROOT / "Reports" / "Manuscript_v2.md"

# (id, why, forbidden pattern, refutation-exempt pattern or None)
RULES: list[tuple[str, str, str, str | None]] = [
    ("criterion-met",
     "The pre-registered criterion was NOT met, and on the reported estimand it was not adjudicable",
     r"criterion\s+(?:was\s+)?(?:is\s+)?met\b|success criterion (?:was|is) met",
     r"not met|never met|fails|was not"),
    ("ci-beside-macro",
     "Quoting a 95% CI beside a country_macro point mixes two estimands; the headline 38.20 sat "
     "outside its own quoted interval",
     r"38\.20.*?\[38\.349|38\.349,\s*104\.537",
     r"outside its own|different statistic|estimand"),
    ("trained-on-dengue-and-influenza",
     "COVID trains the trunk behind every Ebola number",
     r"trained on dengue and influenza|dengue and influenza\b(?![^.]*COVID)",
     r"is wrong|COVID|third (?:development )?disease"),
    ("h10-h15-few-shot",
     "E6 binds h10 and h15 to the zero-shot label on the primary arm; L12 has 0 adaptation pairs at h15",
     r"few-shot at (?:h|\*h\* = )(?:10|15)|(?:h|\*h\* = )(?:10|15)[^.]{0,40}few-shot",
     None),
    ("wins-all-zero-shot",
     "Six of eight confirmed wins are zero-shot; two are L12 few-shot at h15. The safe framing is "
     "horizon-first: all eight are at h10 or h15",
     r"every confirmed win is at (?:h10|h15)[^.]*zero-shot|all (?:eight|8) (?:wins|are) [^.]*zero-shot regime",
     None),
    ("zero-shot-beats-few-shot-tested",
     "Downgraded to an observation; the per-origin bootstrap on the inversion has never been run",
     r"adaptation hurts|zero-shot (?:significantly )?beats few-shot",
     r"observ|not tested|we do not test|cannot be called|no bootstrap"),
    ("90-coverage",
     "Every +ACI figure consumes truth 2 to 14 weeks ahead of real time; label it retrospective",
     r"90% coverage after calibration|achieves? 90% coverage",
     r"retrospective|oracle|ahead of real time"),
    ("no-incumbent-disease-agnostic",
     "epiFFORMA uses the exact phrase for an emerging-pathogen forecaster; needs 'and spatially structured'",
     r"[Nn]o incumbent is disease-agnostic by construction",
     r"spatially structured"),
    ("transfer-not-across-diseases",
     "False on the project's own record (Encoder Decision.md:12)",
     r"across locations for a single disease but not across diseases",
     None),
    ("shap",
     "No attribution code exists anywhere in our source (M12)",
     r"\bours?\b[^.]*\bSHAP\b|we (?:provide|deliver|report)[^.]*\bSHAP\b",
     r"Liu and Cao|incumbent|prior work"),
    ("domain-generalisation-objective",
     "No such objective is in train/; the rejection is on the record",
     r"domain-generalisation objective|domain generalisation objectives",
     None),
    ("mtgnn-12-of-16",
     "MTGNN emits a single constant on 47 of 80 files; beating a constant is not evidence",
     r"better in 12 of 16|12 of 16 against MTGNN",
     None),
    ("91000-step-trunk",
     "Ebola trunks ran 32,000 to 35,000 steps and selected at 2,000 to 5,000; the full-budget "
     "control peaks at step 7,000 and degrades after",
     r"91,?000[- ]step|91,?000 steps",
     r"budget|never|not reached|early|selected at"),
    ("wis-flusight-unqualified",
     "Needs K=2 stated; and WIS and CRPS are bit-identical on this grid, so two columns is misleading",
     r"WIS, the FluSight standard",
     r"K\s*=\s*2|two intervals"),
    ("ldo3-unseen-geography",
     "Hold-out-COVID runs on a graph the trunk saw bit-identically; dengue contains the same 47 "
     "Japanese prefectures as influenza_japan",
     r"holds out a disease on unseen geography|diseases occupy disjoint geography",
     None),
    ("ebola-pcc-smape",
     "Out of protocol against score.py:55-65",
     r"Ebola[^.]*(?:PCC|sMAPE)[^.]*(?:0\.157|111\.2)|(?:0\.157|111\.2)[^.]*Ebola",
     None),
    ("new-cases-cross-check",
     "No such code exists; stated in the present tense as though adopted",
     r"[Nn]ew cases series is adopted as a cross-check",
     None),
    ("all-raw-checksummed-unqualified",
     "True of five of six bundles",
     r"all raw inputs (?:are )?checksummed|nothing is written if any gate fails",
     r"five of six|except"),
    ("stale-mask-density",
     "Superseded; actual 0.4095",
     r"0\.5255",
     r"before|superseded|corrected"),
    ("stale-support-27",
     "Superseded support protocol",
     # \b before the 9, or "59 districts" matches and reports a false positive.
     r"\b27 (?:support )?cells|\b9 districts\b",
     r"superseded|earlier|previously"),
    ("stale-adapter-388",
     "Superseded; the adapter is 1,428 parameters",
     r"388[- ](?:parameter|param)",
     r"superseded|earlier"),
]

MANUAL = [
    ("estimand", "Every Ebola interval must be the district bootstrap from ebola_ci.py, not the "
                 "cell-pooled one from analysis.py. Check each quoted interval's provenance."),
    ("regime-labels", "Any per-horizon Ebola result printed under a few-shot header where E6 binds "
                      "the horizon to zero-shot."),
    ("naive-floors", "Any Ebola win stated without naming which naive floor it beats."),
    ("prereg-language", "Any sentence implying the Ebola result confirms a hypothesis rather than "
                        "reporting a pre-registered test that failed on its own terms."),
]


def scan(text: str, strict: bool = False) -> tuple[list, list]:
    findings, exempted = [], []
    lines = text.splitlines()
    for rid, why, pat, ok in RULES:
        for i, line in enumerate(lines, 1):
            for m in re.finditer(pat, line):
                snippet = line[max(0, m.start() - 60):m.end() + 60].strip()
                if ok and re.search(ok, line):
                    exempted.append((rid, i, snippet, why))
                else:
                    findings.append((rid, i, snippet, why))
    return findings, exempted


def main() -> int:
    strict = "--strict" in sys.argv
    text = DOC.read_text(encoding="utf-8")
    findings, exempted = scan(text, strict)

    print(f"{DOC.name}: {len(text.split())} words, {len(RULES)} mechanical rules\n")
    if findings:
        print(f"=== {len(findings)} UNSAFE CLAIM(S) ===")
        for rid, ln, snip, why in findings:
            print(f"\n[{rid}] line {ln}\n  text: ...{snip}...\n  why:  {why}")
    else:
        print("=== no unsafe claim matched a mechanical rule ===")

    if exempted:
        print(f"\n=== {len(exempted)} passed only via a refutation exemption "
              f"(read these by hand) ===")
        for rid, ln, snip, _why in exempted:
            print(f"  [{rid}] line {ln}: ...{snip[:110]}...")

    print(f"\n=== {len(MANUAL)} items this cannot check mechanically ===")
    for rid, note in MANUAL:
        print(f"  [{rid}] {note}")

    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
