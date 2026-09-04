# What It Would Take to Reproduce This — Stakeholder Brief

**13 August 2026 · Phase 3**

Plain-language companion to the project's reproducibility record. It answers one question in the
form a reviewer will actually ask it: if we handed this project to an independent team tomorrow,
what could they verify for themselves, what would they have to take on trust, and what would it cost
us to shrink the second list. No statistics or engineering background assumed. Everything below was
checked against the working project this week rather than carried over from earlier write-ups.

---

## The short version

**One. Every piece of input data traces back to a named public source, and the build re-checks each
one every single time it runs.** Twenty-five input files across five public sources, each carrying a
recorded fingerprint. If any file has changed by so much as a byte, the build stops rather than
warning and continuing.

**Two. The datasets rebuild from those sources with a single command, and the build refuses to
produce anything that fails a check.** One hundred and one correctness checks run across the six
datasets before anything is written to disk. Six deliberately broken versions are also run, and each
must be caught, because a check that cannot fail proves nothing.

**Three. No number in any report we have sent you was typed by hand.** Every table is generated from
machine-written run records — 449 of them — by a script that reads them. A figure in prose can be
traced back to the record it came from, and that record back to the command that produced it.

**Four. The Ebola case study is protected by construction rather than by discipline.** The protocol
was written and locked before any Ebola number existed, both data configurations were fingerprinted
before either was scored, and the code refuses to score a second time.

**Five. We can state precisely what an outside team could not reproduce today, and what each gap
costs to close.** The largest is that our run outputs are not published alongside the code, so a
reviewer would have to re-run tens of hours of computation to rebuild our tables. Publishing the 449
files that actually matter is a small job, and it is the highest-value item remaining.

---

## One: where the data comes from, and how we prove it has not moved

**The reproducibility problem with public health data is silent drift.** Surveillance datasets get
re-issued, corrected and re-cut. A file downloaded today may not be the file that was downloaded
last year under the same name, and nothing about it announces the change. If that happens
unnoticed, every figure in the paper shifts while every internal check still passes, because the
checks test the data against itself.

**So each input carries a recorded fingerprint, and the build verifies all of them before it starts
work.** Twenty-five files: the dengue extract, the Ebola compilation, six influenza benchmark
matrices, the COVID-19 series, and sixteen map files. A mismatch halts the build with the expected
and actual fingerprints printed side by side. There are no exceptions and no warnings, because a
warning in a long build is a message nobody reads.

**Two of the five sources needed particular care, and both are worth understanding.**

The Ebola compilation is the only source with no version number at all, so its fingerprint is its
only version identifier. There is a real story attached: an early working copy of that file had been
opened and re-saved in a spreadsheet application, which rewrites the file's container without
altering a single number. Its contents and the dataset built from it were verified identical — but
its fingerprint no longer matched the published file, so a reviewer checking their own download
would have got a mismatch against data that was in fact correct. The pristine published file is now
the source of record.

The COVID-19 series is downloaded from a location that can move by construction, even though the
publisher archived the repository in 2023. That is precisely why it is fingerprinted rather than
exempted. A source that arrives by download is *more* exposed to drift than one placed by hand, not
less, so exempting it would have been backwards.

**The datasets themselves ship with the code.** Six built datasets plus the two frozen Ebola
configurations, sixteen megabytes in total, in a plain array format that any tool can read. That
means a reviewer can check every result without first reconstructing a gigabyte of downloads. The
raw sources sit in a separate data store rather than in the code repository, for size and licensing
reasons, but every one of them is publicly obtainable and we publish the address and fingerprint of
each.

---

## Two: the build refuses to produce anything that fails a check

**One command rebuilds all six datasets from the raw sources.** It verifies the inputs, constructs
the datasets, runs the full correctness suite, and only then writes anything.

**The order is the point.** The checks run *before* the write, not after it, so the only dataset
that can exist on disk is one that passed. This is a deliberate response to how data defects
actually escape: a dataset containing a flaw, once written to disk, is a dataset somebody will train
on, and the flaw then propagates into results that look fine.

**One hundred and one checks, across all six datasets.** They cover the properties that would
silently invalidate the work rather than crash it: that no normalisation statistic is computed from
data the model is not allowed to see, that no feature is derived from a future value, that the
quantity being forecast never appears among the inputs, and that the training, validation and test
periods are ordered and do not overlap.

**Six deliberately broken versions are run alongside them, and each must be caught.** This is not
belt-and-braces; it is the lesson of a real failure. An earlier version of our suite recovered the
original case counts by reversing the very transformation it was testing — circular reasoning — and
it passed a deliberately corrupted Ebola dataset with every check showing green. Each dataset now
carries its untouched raw counts, every check is evaluated against those, and the planted faults
confirm on every run that the checks are still capable of failing.

**The count rose from 86 to 101 during this reporting period.** COVID-19 was previously built by a
separate script, which meant it sat outside the checks entirely. Folding it into the single build
command brought it under the same fifteen per-dataset checks every other dataset passes, and all
fifteen pass. We verified the rebuilt COVID dataset is identical to the released one first, so no
COVID number anywhere in the project moved.

---

## Three: no reported number is typed by hand

**Every run writes machine-readable records, one per model, dataset, horizon, random start and
measure.** There are 449 such record files. Every table in every document we have sent you is
produced by a script that reads them and formats them.

**This is the property that makes the paper auditable.** A claim in prose traces to a record, and
that record to the command that wrote it. It also means our tables cannot disagree with our data
through a transcription slip, which is the most common and least detectable way a results section
goes wrong.

**Four reporting rules are built into those scripts rather than left to the person writing.** Each
was coded because it had been violated at least once first. Every figure carries its run-to-run
spread and the number of runs behind it, and a single-run figure is labelled as one rather than
dressed up as an average. Results are never averaged across datasets or across the two Ebola
configurations, because those are different questions rather than repeats of one. Every comparison
states its reference point in the table's own heading. And a difference whose interval covers zero
prints as "within noise" rather than being given a direction.

**One of these scripts refuses to print at all under a specific condition.** The Ebola reader checks
that all twenty run records agree on the protocol fingerprint and the support-set definition. If
they disagree, the run mixed configurations and the table would not be a single experiment, so it
declines rather than averaging across them.

---

## Four: the Ebola result is protected by construction

**The case study is scored exactly once, which changes what reproducing it means.** For every other
result, reproduce means re-run. For this one it does not, and the difference is deliberate.

**The protocol was locked before any Ebola number existed.** It fixes the training setup, the
adaptation rule, the evaluation window, the measures, the benchmarks and the expected outcomes. Both
support-set configurations were built and fingerprinted before either was scored, and those
fingerprints are recorded in the protocol document.

**Three mechanisms enforce this in code rather than by memory.** The training script checks the
frozen fingerprints before it will score anything, and stamps the protocol fingerprint into every
record it writes. The overnight scheduler refuses to start the Ebola step if scored results already
exist, with a deliberate override that has no accidental path to it. And Ebola never enters any
decision: every setting was chosen on the other diseases, and Ebola carries no validation period by
design.

**So reproducing the Ebola result means re-deriving the published figures from the archived records,
which takes seconds.** A second scoring run would not be a replication. Under the pre-registration
it would be a protocol violation that quietly overwrote the record of the first.

---

## Five: what "reproducible" means for the model runs, precisely

**Three different things are being claimed by that one word, and they deserve separating.**

**The data build is exactly reproducible, and we check it.** A built-in option builds every dataset
twice and compares the results element by element. They match.

**The model runs are reproducible on the same machine, and we have measured that rather than assumed
it.** Each run is started from a fixed random seed. When we re-ran three transfer experiments under
a changed training budget, all eight resulting records came back identical to the originals — which
is also the check that settled a separate question about early stopping.

**They are not guaranteed identical on different hardware, and we say so.** Graphics cards from
different generations, or different versions of the underlying numerical libraries, can add numbers
in a different order and produce slightly different results. We have not forced the strictest
determinism setting available, because doing so would slow every run substantially to protect a
property that no external reviewer on different hardware would get anyway.

**What carries the conclusions instead is repetition.** Every headline is reported over five fixed
random starts with the spread shown, and every comparison is paired so that the shared start-up
noise cancels. A conclusion that survives that treatment does not depend on any individual run
matching to the last decimal place — which is the property we actually want, since the reviewer's
numbers will differ slightly from ours no matter what we do.

---

## Supporting evidence

**Ten independent checks can be run on a fresh copy in minutes, without a graphics card**, and each
one fails loudly rather than quietly. They cover the correctness suite and its planted faults, the
34 unit tests of the data transformations, two tests that confirm properties of the real source
files that our loaders depend on, the architectural constraints that make the model
disease-agnostic, the measurement layer, the analysis engine and the uncertainty calibration logic.
The last two are tested on synthetic data with known answers, so they verify the method rather than
the result.

We re-ran all ten while preparing this brief. All pass.

---

## What an outside team could not do today

**Our run outputs are not published with the code.** This is the significant one. A reviewer gets the
code and the datasets but not the results, so every table would have to be regenerated by re-running
the experiments — tens of hours on a graphics card. The full set is 1,248 files and 3.7 gigabytes,
but the 449 record files that every table is actually built from are a small fraction of that.
Publishing those is a modest task and would let a reviewer reproduce every table in the paper in
seconds.

**The eight comparison models are not bundled.** They are other teams' published code, run in place
in their own environments. We orchestrate and score them, and we document each model's published
claims, but a reviewer would have to obtain the eight themselves. Two of the eight remain
problematic and are reported as such rather than papered over: one cannot be run on our hardware at
all, and one has an unresolved discrepancy between its published table and its own shipped data.

**Three recent pieces of work are not yet under version control.** The epidemiology-informed
component, its evaluation, and the look-back window sensitivity study are complete and their results
are in hand, but the code sits outside the tracked project. This is housekeeping, not a research
gap, and it should be closed before the package is assembled.

**Runs do not record how long they took.** The timings we quote come from notes in the scheduling
scripts rather than from the records themselves, so "how long would this take to reproduce" is a
question we can only answer for some of the experiment families. Worth fixing, cheaply, before the
package ships.

**The exact software environment is pinned to the version but not to the exact build.** We publish a
complete list of every package and version used. The stricter file that would let someone recreate
the environment down to the individual build identifier exists but is not published.

**Some older internal documents carry superseded counts.** Two of them still say five datasets and
86 checks; the current figures are six and 101. The reproducibility record now states which document
is authoritative where they disagree, but the stale ones should be corrected before external release.

---

## Where this is weak

**The reproducibility claim rests on one machine.** Everything reported was produced on a single
graphics card under Windows. Nothing requires that specific hardware and the datasets rebuild
anywhere, but we have not actually run the training on a second machine, so "reproducible on
different hardware" is a reasoned expectation rather than something we have demonstrated.

**Access to the raw sources is not a single step for an outsider.** Our own copies live in a private
data store. Every source is publicly obtainable and we publish where and how, but reconstructing the
inputs is several downloads rather than one command. That is a licensing and size constraint rather
than a choice, and the fingerprints mean a reviewer can confirm they ended up with the right files.

**A reviewer cannot independently reproduce the Ebola scoring run,** by design. They can re-derive
every published Ebola figure from the archived records, and they can verify that the protocol was
fixed in advance and that the scored data matches its recorded fingerprint. They cannot watch it
happen. This is the correct trade — running it twice would destroy the property that makes the
result credible — but it is a genuine limit and it should be stated in the paper rather than left
for a reviewer to notice.

---

## Where this leaves the milestones

The internal brief's definition of done lists the manuscript and its reproducibility package as one
of the two outstanding conditions, alongside the ablations and explainability. This document and the
technical record behind it are the first component of that package, and they are complete: the
build, the checks, the commands, the environments, the pre-registration protections and the honest
limits are all now written down in one place rather than distributed across a dozen documents and
several people's memory.

What remains for the package itself is a short, well-defined list rather than open-ended work:
publish the 449 record files, bring the three recent pieces of code under version control, correct
the superseded counts in the two older documents, and add run timings. None of these is a research
task and none blocks the remaining science.

---

## What happens next

The record files are the item worth doing first, because publishing them converts our results from
something a reviewer takes on trust into something they can check in an afternoon — and reviewers of
this kind of paper increasingly expect exactly that. The remaining items are housekeeping and can be
folded into the package assembly at the end of the extension period.
