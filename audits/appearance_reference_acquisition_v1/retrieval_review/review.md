# Archived appearance-reference retrieval review

Date: 2026-10-10. Reviewer: `appearance_archive_retrieval`, a separate-context AI subagent. This is a bounded acquisition review. No participants or authors were contacted. No model experiments, stimulus reconstructions, or dataset qualification were performed.

## Outcome

**No participant rows, stimulus source files, or original archive bytes were acquired.** Public metadata now identifies a concrete, documented CSV download rather than a guessed endpoint. Its retrieval fails in this environment. Availability metadata is not evidence that a dataset has been inspected or qualified; it does not remove the file-level gate.

The first download to attempt when public data access works is the maintainer's [metacognition CSV](https://mspeekenbrink.r-universe.dev/sdamr/data/metacognition/csv), linked by the [maintainer dataset index](https://mspeekenbrink.r-universe.dev/datasets). The index describes a 7,560-by-10 table. A [JSON export](https://mspeekenbrink.r-universe.dev/sdamr/data/metacognition/json) is independently linked there. Both web-tool clicks return `Cache miss`; a direct CSV request fails with temporary DNS resolution failure. Neither returned rows.

## Documented provenance lead

The maintainer's [package API](https://mspeekenbrink.r-universe.dev/api/packages/sdamr) was readable through the web tool. It identifies package `sdamr` version `0.2.0`, the `metacognition` table and its columns: `id`, `age`, `sex`, `block`, `trial`, `tilt`, `contrast`, `correct`, `visibility`, and `confidence`. The API advertises 7,560 rows, upstream revision `93245a342bf0d6c5c4cf26d9cab124b67920e51c`, and source archive `sdamr_0.2.0.tar.gz`. Its advertised archive size is 321,468 bytes and SHA-256 is `8639e83665ce170e60224ab9b5b16f9853ed7bf90e31c382e61f96d427f87e45`. These values describe remote metadata; no locally acquired archive was hashed against them.

The package page links its [source archive](https://mspeekenbrink.r-universe.dev/src/contrib/sdamr_0.2.0.tar.gz); the API also supplies a [content-addressed archive URL](https://r2.ropensci.org/8639e83665ce170e60224ab9b5b16f9853ed7bf90e31c382e61f96d427f87e45). Both remain inaccessible. The API declares package GPL-3. That declaration does not independently verify the original participant-data license, author-archive permissions, stimulus-code license, or exact redistribution transformations. No license was inferred from article access.

The [maintainer manual](https://mspeekenbrink.r-universe.dev/sdamr/doc/manual.html) attributes the table to [OSF vk6fe](https://osf.io/vk6fe/). It describes practice exclusion and numeric orientation, task-correctness, visibility, and confidence fields. Its rating endpoints are incompletely rendered. Missingness, exclusions, trial uniqueness, participant count, rating order, stimulus shape, exact contrast values, and transformations must still be checked from acquired files, not assumed from documentation. The documented schema lacks explicit shape and rating-order columns.

## Stimulus source boundary

The [primary article](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2016.00591/full) describes MATLAB/Psychtoolbox presentation, small circular or square binary gratings, six contrast levels, horizontal/vertical orientation, and 200-ms presentation. It collects clarity and response-confidence ratings on each trial, with rating order counterbalanced across participants and subsequent error feedback. That is a useful empirical dissociation target, not an attention-selection manipulation or free-prose reference corpus. The [author archive landing page](https://osf.io/vk6fe/) advertises full material but does not expose inspectable file content here. Exact grating generation, monitor calibration, per-trial stimulus shape, archived program bytes, and trial-level reconstruction remain unverified.

## Bounded attempt record

Successful reads: maintainer package page, dataset index, package API, manual, primary article, and author archive landing-page text. Failed web retrievals: official CSV and JSON exports, package archive, package DESCRIPTION, content-addressed archive, CRAN-mirror CSV, and maintainer NDJSON export. Direct requests for CSV, source archive, content-addressed archive, and OSF node file listing all fail with `[Errno -3] Temporary failure in name resolution`. An authenticated `gh api` read of the plausible upstream `data/metacognition.rda` path fails connecting to `api.github.com`; the path's existence was consequently not established. The shell lacks a `python` executable; direct requests were rerun with the project's `.venv/bin/python`. No security boundary was bypassed.

Stop here rather than manufacture records or substitute scale labels for participant observations. The next permitted acquisition can use the documented CSV or archive URL, pin actual bytes, verify licenses and provenance, inspect participant/trial schema, and then qualify a participant-held-out analysis. A reconstruction from the paper must be labeled reconstruction and cannot replace missing native stimuli or archive code. This acquisition failure is an external access limitation; it is not evidence against the existence of the data or the consciousness theory.
