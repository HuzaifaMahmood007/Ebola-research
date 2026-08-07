I've been through the Week 3 summary properly. Good work overall, and I've made the direction call you asked for at the end. I've put the whole thing in this email rather than a separate memo so it's all in one place.

Starting with the time sensitive bit.

Hold the five seeds for now

You're right that transfer is the direction. That's the paper and I'm backing it. But I don't think our LODO folds are measuring what we're claiming they measure.

Three of our four datasets are influenza. So when we hold out flu-Japan, the shared encoder has still trained on flu-US-regions and flu-US-states. We haven't held out the disease, only the population.

Look at where that bites. Our two strongest results, US-regions at \+20 to \+28% and Japan at \+24%, are the folds with the most flu leakage. Dengue is the only genuinely held out disease we have, and it's the fold where transfer barely moves the needle. That pattern is suspicious and a reviewer will read it the same way.

I'm not calling the run wrong. This is a consequence of the dataset mix we ended up with and I own that as much as you do. But if we spend five seeds on the current fold structure, the compute is gone and we still won't have the number the paper needs. So fix the folds first, then spend the seeds.

Please report both structures as separate tables:

Leave-one-dataset-out, which is what you have now. Keep it. It's a real result about population transfer and it's worth having.  
Leave-one-disease-out, holding all three flu datasets out together as a single fold, with dengue as the other. This is the honest cross-disease number and it's the one our central claim rests on.  
I expect the honest number to come in weaker than the current headline. That's fine. A modest result that survives review is worth a lot more than a strong one that gets taken apart, and having both tables side by side actually lets us say something specific about where the transfer is coming from. That's a better paper than either table alone.

Your three options

Transfer, yes, with the fold fix above.

Shoring up single-disease accuracy first, no. We aren't claiming a better single-disease model. The thing that protects us from "you only beat your own weak baseline" is the published baseline head to head, not internal tuning. Two exceptions and I want both kept tight. Seed ensembling is close to free so take it, lower variance makes every comparison in the paper easier to defend. Delta prediction gets a quick test on one dataset and one seed, then bring me the number and we'll decide. If it doesn't win clearly, drop it and write me a paragraph on why. That paragraph is useful in the paper either way.

The baseline benchmark isn't really a third option. It's the Week 3 gate. Nothing gets signed off without it, so it's top priority.

Numbers that need fixing before the next run

The percentages in your transfer section don't reconcile with the single-disease table. US-regions h3 goes 613 to 553, which is a 9.8% improvement, but the table says \+20%. US-states h3 goes 113 to 112, which is under 1%, but says \+6%. Japan h5 goes 841 to 779, about 7.4%, but says \+4%.

My guess is that the single-disease table is a 5-seed mean while the transfer table compares against that same seed's own run. If that's it, fine, but it isn't stated anywhere and as printed the two tables can't be read against each other. Label the comparison reference directly on the table.

Separately, dengue h3 and h5 look sign inverted. 42 to 49 at h3 is a 17% worsening but it's printed as \+3% under a "positive is better" convention. Same shape at h5. Even a seed matched reference would have to sit implausibly far from the mean to produce that. Please check the delta computation directly, in case it's affecting other cells too.

Also, on the joint training table, please stop drawing conclusions from cells under about 5%. Deltas of plus or minus 1% across five seeds with no standard deviations reported aren't distinguishable from seed noise, and right now the table invites a reviewer to point that out. And the "spatial contribution" column needs units and a formula, one sentence covers it.

Reporting standards from here

Rather than retrofitting this in Week 6, I'd like these to apply to everything from now on:

Every number gets a dispersion figure. Mean and standard deviation at minimum, bootstrap confidence intervals over regions and time origins for anything we call a headline result. No more bare point estimates.

Never average across datasets. You're already doing this right, just keep it. Dengue at 7,165 regions and US-regions at 10 can't go into the same mean.

State the comparison reference on every delta table, on the table itself.

If a cell doesn't clear the noise floor, write "within noise" rather than describing a direction.

Ebola audit, please expedite this one this week

This is the single thing I most need from you and I don't think the Week 2 audit note ever reached me. It's now the biggest open risk on the project.

What I need in writing: exact temporal resolution, total usable time origins after the support and query split, missingness across the 61 units, whether we have a usable adjacency graph for those units, and most importantly, how many labelled examples the Ebola support set actually gives us.

Run the arithmetic out loud for me. At 61 regions and roughly a year of data, a 20 week input window plus a 15 week horizon eats 35 weeks per sample. If that data is weekly we might be looking at something like 17 usable time origins before we split anything. If that's right then h10 and h15 may not be meaningfully evaluable on Ebola at all, and that matters enormously because far horizon is where transfer currently underperforms. We'd be landing our case study in the one regime the method is weakest in.

One more thing on this. The encoder memo set the few-shot design floor at 27 labelled examples, taken from the smallest development set. If Ebola comes in under that number, we calibrated the constraint against the wrong thing. I'd much rather know that now than once we're into the Ebola work.

A gap I want to talk about: meta-learning

What's implemented, train on three, freeze, fit an adapter, is transfer learning with a linear probe. It's a sound approach and it has a real virtue, which is that it's exactly how the Ebola step will work. But it isn't meta-learning, and the brief has meta-learning and domain generalisation as REQUIRED under G2. We also chose this encoder specifically because it supports cheap second order differentiation, and that argument only pays off if we actually use it.

My call is to build episodic meta-training across diseases, MAML or Reptile or ProtoNet family, your pick, just tell me which and why. Freeze-and-adapt then becomes the ablation. That's a better structure than either on its own because it lets us answer "does meta-learning buy anything over a linear probe" with a number instead of an assertion.

Scope it this week and tell me straight away if you think it doesn't fit the schedule. If it doesn't, I'll take the re-scope to Nora. Please don't quietly absorb it.

Rest of the work order

Once the above is moving, in rough priority:

Finish the baseline head to head, EpiGNN, ColaGNN and the rest of the locked shortlist, on our pipeline, all four datasets, all four horizons. Two conditions I want stated in the output. First, validate each reproduction against the source paper's published numbers before using it as a comparator, and show me our reproduction next to their reported figure. If we can't get close, that baseline doesn't go in the comparison table. Second, start the reproduction failure log now, every baseline we couldn't reproduce and exactly why. You were right in the encoder memo that this is publishable on its own, and it's far easier to keep as you go than to reconstruct at the end.

Joint training, give it two days and then stop either way. Your diagnosis is right, dengue is about 98% of training examples so the shared model becomes a dengue model that glanced at flu. But we haven't tried the obvious remedies and a reviewer will ask whether we balanced the sampler. Try balanced or temperature scaled sampling, per-disease loss reweighting, per-dataset normalisation. If it still fails after that, the negative result gets much stronger. It stops reading as an unfinished experiment and becomes a clean motivation for freeze-then-adapt.

Add the metrics this field actually evaluates on. RMSE, MAE and PCC won't satisfy reviewers from the epidemic forecasting community, and we're claiming calibrated uncertainty. We need WIS, which is the CDC FluSight and Forecast Hub standard, plus CRPS, empirical coverage at nominal levels and PIT histograms. Peak timing and peak intensity error too, epidemiology reviewers care about those specifically. And a scale normalised error alongside raw counts, because across 7,165 dengue regions raw RMSE is dominated by the biggest ones.

Add ARIMA or SARIMA and a gradient boosted model on lag features to the simple baseline set. The GBM in particular is hard to beat and I would much rather find that out from you than from a reviewer.

Reserve the conformal calibration split in the pipeline now. Conformal is the UQ approach, the encoder build settled that by omission and I'm confirming it formally. But use a shift and time aware variant, not textbook split conformal. Vanilla split conformal assumes exchangeability and our setup breaks that on two axes at once, temporal dependence and distribution shift onto a disease the model has never seen. Applying standard split conformal to a transferred model on Ebola is a hole that specialists will find immediately. Look at adaptive conformal inference, conformalized quantile regression, or EnbPI, with a weighted variant for the covariate shift. Pick one and tell me which, because the calibration split has to be reserved before we get into the Ebola work and retrofitting it is painful.

Scope explainability now rather than discovering it later. SHAP over a spatio-temporal GNN with 7,165 regions is a serious compute proposition, and with only four input channels it isn't obvious what we're attributing over, channels, time lags, or neighbouring regions. Integrated gradients is probably more tractable. Come back with a scoping note. And separately, the learned graph gate is already an interpretable result and we're under-using it. Gate values of 0.27 to 0.60 scaling with region count and density is a figure. Build it.

Pre-registration, before Ebola gets touched. Freeze the config, hash it, timestamp it, commit it, then score the Ebola set exactly once against it. This was your recommendation in the encoder memo and I'm adopting it as a stated methodological contribution rather than an internal process note. It's cheap, it's unusual in this literature, and combined with the reproduction failure log it gives the paper a credibility story that most submissions in this space can't tell.

What I'm explicitly deprioritising

Flu-Japan losing to seasonal naive. Your diagnosis is correct and the explanation you wrote is good enough to publish as it stands. Do not add a t-52 lag channel to the main model. It would fix the benchmark row, but Ebola has no prior year, so that channel would be permanently null for our actual target. That's a train and target mismatch introduced to win a table cell. If we want it at all it runs as a benchmark only variant with the caveat stated, and that's a decision for later.

Single-disease accuracy work beyond the two exceptions above.

The epidemiology informed component stays as a planned ablation, not main build. The brief tags it as a suggestion so this is legitimate, but I'm confirming it as a decision rather than letting it drift.

What I'm taking to Nora

Adding COVID-19 back as a third development disease. It's in the brief as a suggestion and it got dropped. Restructured folds make our claim honest, a third disease makes it strong, and I want it back.

The Ebola horizon set, pending your audit. If the data says near term only, I'd rather narrow deliberately and justify it operationally than be quietly weak at h15.

Encoder confirmation, which has been provisional since 20 July with everything downstream assuming it.

Baseline shortlist sign off. The brief requires client sign off before we lock, and I can't tell from our records that it happened.

Schedule, journal, authorship and IP.

You aren't blocked on any of those, so keep moving.

To summarise what I'd like expedited this week

The Ebola audit numbers, the reconciled transfer table, your read on whether meta-learning fits, and the baseline head to head. Once the folds are corrected the five seed run can go straight after.

Last thing

The feasibility section is the best work on this project so far. Pairing every check with a deliberately broken version that has to fail is not something most teams bother doing, and it's the reason I'm willing to trust the accuracy numbers at all. It's going in the paper as a stated contribution, not an internal note. Same for the score-Ebola-once idea and the reproduction failure log from your encoder memo, both adopted.

And flagging the single seed caveat yourself, without being asked, is exactly what I want from you. I'd rather hear a result is fragile from you than find out in review. Keep doing that.

Give me a shout if any of the above doesn't sit right, particularly the meta-learning scoping.