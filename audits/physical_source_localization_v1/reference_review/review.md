# Independent appearance-data discovery review

Date: 2026-10-10. Reviewer: `appearance_reference_discovery`, an AI subagent working in a separate context. This is a literature and data-availability review, not human validation, a completed dataset qualification, or a claim that subjective experience has been measured in the synthetic system. No participants were contacted and no files elsewhere in the repository were changed.

## Recommendation

The strongest small candidate is **Rausch and Zehetleitner (2016)** for a future assay separating reported visual clarity from decision confidence and task accuracy. **Anderson (2016)** is a complementary candidate for contrasting apparent-contrast reports with discrimination and selection. Neither has yet passed file-level qualification here. Preserve the synthetic physical-source study as an initial component: physical-feature decodability or invariance alone cannot establish correspondence with participant appearance judgments.

## Candidate 1: same-trial clarity, confidence, and discrimination

[Rausch and Zehetleitner, Frontiers in Psychology, DOI 10.3389/fpsyg.2016.00591](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00591/full) describes 20 participants, nine blocks of 42 trials, six physical-contrast levels, horizontal/vertical discrimination, and separate continuous clarity and confidence judgments on every trial. Rating order was counterbalanced; feedback followed the ratings. These are empirical participant judgments rather than a collection of suggested scale labels. They are not free prose, brightness/contrast estimates, or a manipulation of attentional selection. Simultaneous ratings, feedback, and question order limit generalization.

The paper identifies its [author archive](https://osf.io/vk6fe/). Its public landing page, authored by Manuel Rausch and Michael Zehetleitner, describes full material and displays 3.8 MB. The file listing and trial files could not be retrieved. Stimulus-generation code and exact stimulus reconstruction remain unverified.

A [CRAN-mirror package manual](https://www.stats.bris.ac.uk/R/web/packages/sdamr/refman/sdamr.html) documents a redistributed `metacognition` table with 7,560 rows and columns `id`, `age`, `sex`, `block`, `trial`, `tilt`, `contrast`, `correct`, `visibility`, and `confidence`. This is a retrieval lead, not an inspected or checksummed copy of the author's raw data. It does not document per-trial images or selection.

## Candidate 2: contrast matching versus discrimination/selection

[Anderson, PLOS ONE, DOI 10.1371/journal.pone.0152353](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152353) uses checkerboard stimuli, uninformative spatial cues, continuous contrast variation, and different task instructions. Participants reproduce orientation or contrast with response meters. Experiments reverse which attribute selects the response side and which is reported; later comparisons expose a difference between inferred contrast appearance and direct discrimination reports. This offers a useful challenge to treating cue preference as a sufficient appearance measure. Trial confidence is not documented. Different experiments use different participants, and feedback and selection can influence reports.

The primary article identifies data and analysis scripts at [OSF tnp9h](https://osf.io/tnp9h/). The landing page, API listing, and files could not be retrieved here. Stimuli are described procedurally, but archived generation code, exact file columns, licenses for the data, and the raw participant-level records remain unverified. This candidate cannot yet be enrolled in a held-out prediction assay.

## Candidate 3: color-similarity warping with selection

[Feature-based attention warps the perception of visual features, Scientific Reports, DOI 10.1038/s41598-023-33488-2](https://www.nature.com/articles/s41598-023-33488-2) combines oddball-color search with similarity judgments. One control uses balanced distractor colors and varies distractor salience. This could support an appearance-versus-selection benchmark, but it concerns continuous color similarity rather than the apparatus's four-category identity output. Trial confidence is not documented. The authors state that full experiment data and analysis scripts are at [OSF w9kc5](https://osf.io/w9kc5/). OSF landing/API retrieval failed, so raw columns, stimulus code, and downloadable participant data are unverified.

## Methodological constraint from a primary contrast study

[Itthipuripat and colleagues, Journal of Vision, DOI 10.1167/19.14.8](https://pmc.ncbi.nlm.nih.gov/articles/PMC6908139/) independently varies cued and uncued contrast, uses orientation responses in a comparison task, and adds an equality task and blank-stimulus trials. The authors distinguish cue-driven baseline response bias from proposed appearance effects and show that inferred effects depend on judgment method and uncertainty. A point-of-subjective-equality shift alone is therefore a weak benchmark. The article was inspected for assay design; no raw-data repository was verified, so it is not being represented as an available participant dataset.

## Availability and qualification boundaries

Only article and landing-page text was retrieved. **No trial dataset, stimulus archive, source tarball, participant prose, or analysis code was downloaded.** Shell attempts to the OSF APIs for `tnp9h` and `w9kc5` failed with temporary DNS resolution errors. Web retrieval of their pages/APIs returned internal errors. `vk6fe` landing-page text worked, but its Files tab and API did not. CRAN package index/source-tarball and a possible R-universe JSON endpoint were also inaccessible through the web tool. These are observed access failures, not evidence that the archives do not exist.

Qualification should first pin actual file bytes, provenance, license, participant IDs, trial schema, exclusions, rating definitions, and stimulus code or sufficient rendering parameters. Reproduce a source-paper dissociation using independently held-out participants; split by participant before fitting and preserve the native scale rather than generating target prose. Verify whether response-order and stimulus-type covariates exist in the raw archive, since the redistributed table's documented fields omit them.

Then predeclare how source stimuli enter this apparatus. Its RGB 16-by-16 categorical visual encoder and allocation/quality/command attention model do not automatically reproduce a calibrated human contrast task. Downsampling, background conversion, grating frequency, and stimulus duration can alter the relevant signal. Contrast identity and physical intensity must remain separate from participant judgment. A trained readout should be compared with visual-encoder and category/access/history baselines and evaluated on new participants and stimuli. A property predicted only by upstream visual features cannot be attributed to the attention-control model by reporting it through a bound-state interface.

These numerical judgment datasets could supply an operational correspondence target. They cannot validate human-like free-report language, prove felt appearance, certify an appearance head by its name, or equate recovery forecasts with phenomenal clarity. Appearance-related correspondence remains a separate unresolved component after the physical-source study.
