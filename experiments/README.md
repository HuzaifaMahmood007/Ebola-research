# experiments/

Exploratory, off-protocol runs. **Nothing in this directory is a scored result.**

Every run here is labelled `EXPLORATORY` in its own records. None of it may be quoted as the Ebola
case-study result, compared against the pre-registered record in a document without that label, or
used to select anything that later touches the frozen arms. The pre-registered Ebola record lives in
`results/ebola/` and was scored once under `progress/decisions/Ebola_Prereg.md`; changing which
normalisation the Ebola arm is scored under is a protocol change that would need client
re-registration.

Scripts here never write into `results/` or `data/`. The normalisation probe verifies the frozen arm
hashes against `configs/ebola_arms.json` before and after it runs, and mutates bundles in memory
only.
