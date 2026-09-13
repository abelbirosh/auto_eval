---
version: 1
slug: "auto-eval-static-dashboard-html"
primary_target: "auto_eval/static/dashboard.html"
related_targets: ["auto_eval/static/index.html"]
---

# Surface brief — Auto-Eval web UI

Scope: `auto_eval/static/dashboard.html` (runs) and `auto_eval/static/index.html`
(classifier). Visitor mode: **Operate**.

Audience and job: someone choosing between providers who needs a defensible
comparison. They describe what they want compared, then read a result they must
be able to act on and defend. Constraint: local server, no auth, `--mock` path
must look identical minus live numbers. Binding user constraints: blocky, hard
90° edges, no rounded corners, no glass or translucency, dark ground.

## Direction contract

THESIS: A run is a calibration certificate — this instrument was measured,
against this reference, on this date, to this uncertainty. The surface owns
traceability: no figure appears without the conditions that produced it. It
refuses the eval-dashboard default of stat tiles over a donut over a table,
where the headline floats free of its method.

OWN-WORLD: Engraved dark instrument plate, not paper. Tinted near-black ground,
one cooler panel layer, warm bone ink. Reversed solid blocks (knocked-out type)
carry every key figure; hard 1px rules build the certificate grid; zero radius,
zero gradient, zero blur anywhere. Brass is the seal and the only accent, used
for primary action and current selection. Green/amber/red are reserved for
passed/blocked/failed and never decorate. One type size across tabular data —
rank is weight, case, reversal, and rule.

STORY: The reader understands what was measured and what was not, believes it
because the conditions are printed at the same authority as the grade, and acts
by running the suite or opening a case.

FIRST VIEWPORT: A stamped header block — certificate number (run id), instrument
(model), reference (suite + hash), date — as a hard-ruled provenance row. Beneath
it, the result plate: pass rate as a reversed brass-ruled block at display scale,
its denominator set beside it, and immediately adjacent at equal weight the
disclosure strip naming what did not run and why. Then the deviation table, one
size, hard rules, tabular numerals. Primary action ("Run suite") sits top-right
of the header block as a solid brass slab.

FORM: Calibration certificate — candidate 1 of my grounded list, chosen by the
user over the roll's assignment (candidate 3, Regulated Comparative Label).
Seed key 63268c0f.

Carried donations, named: flat palette with no gradient or soft shadow
(tdr-info-noise-sleeve); one type size for tabular data (rw-timetable-slide-rack);
reserved caution vocabulary and every figure showing what it excludes
(signals-instruments-night-flight-six-pack); the grid holding at density with
zero ornament (ikeda-datamatics); one always-level legend owning the three-state
vocabulary (gravity-rain-garden); one primary control per surface with locked
named states (drawcord-transforming-cape).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Unresolved

Contamination display on the dashboard is currently a badge; whether it earns a
column in the deviation table is open until the table is built with real runs.
