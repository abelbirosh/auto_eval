# Design

Auto-Eval's web surfaces, documented from the built world.
Direction contract: `.impeccable/surfaces/auto-eval-static-dashboard-html.md`.

## World

**Calibration certificate, engraved on a dark instrument plate.** A run is a
measurement — this model, against this suite, on this date, to this uncertainty.
The surface owns traceability: no figure appears without the conditions that
produced it. It refuses the eval-dashboard default of stat tiles over a donut
over a table, where the headline floats free of its method.

Hard 90° edges everywhere. No border-radius, no gradient, no blur, no
translucency, no box-shadow. Rank inside tabular data is carried by weight,
case, reversal and rule — never by a change of type size.

## Tokens

| Token | Value | Role |
|---|---|---|
| `--ground` | `#0D1012` | page ground (tinted, never pure black) |
| `--panel` | `#151A1D` | second neutral layer: form beds, toolbars |
| `--plate` | `#1E2429` | engraved block: open rows, code wells |
| `--rule` | `#2B3238` | hairline grid |
| `--rule-2` | `#414B52` | structural rule, field underlines |
| `--ink` | `#EDE8DF` | primary ink — 15.65:1 on ground |
| `--ink-2` | `#A8A79E` | secondary ink — 7.90:1, tinted from the ink, never neutral gray |
| `--pass` | `#48A260` | reserved status: passed |
| `--blocked` | `#3E92D6` | reserved status: could not be checked |
| `--fail` | `#DC6E5E` | reserved status: failed |

The three status colors were computed, not chosen. They clear the OKLCH dark
lightness band (0.48–0.67), the chroma floor, adjacent-pair CVD separation, the
normal-vision floor, and ≥4.5:1 text contrast on all three surfaces. Re-validate
with the dataviz validator before changing any of them; an earlier amber for
`blocked` failed the normal-vision floor against `fail`, and a teal `pass` read
as the cyan-on-dark AI tell.

**Status colour is never the only carrier of a verdict.** Every verdict is also
spelled out in caps, and a fixed always-level legend defines the three words.

## Type

One family (system sans) for UI; monospace for data, identifiers and
measurement — earned, not costume. One size (13px) across tabular data.
Body 14px. Section labels 11px caps, tracked .18em. Headline grade 58px.
No functional text below 11px. Prose carries a 62ch measure; data tables
deliberately do not.

## Components

- **Masthead** — title reversed above a 2px ink rule.
- **Provenance stamp** (`.stamp`) — certificate / instrument / reference suite /
  spec / judge / measured, in monospace, ruled into equal cells.
- **Result plate** (`.certify`) — the grade reversed out of solid ink, and
  directly beside it at equal authority the disclosure strip: what did not run,
  and why. The disclosure is never beneath the grade and never smaller.
- **Legend** (`.legend`) — fixed, always level, the only place the three words
  are defined.
- **Tally bar** (`.bar`) — square ends, 2px ground gap between segments, per
  segment hover.
- **Primary action** — solid ink slab with knocked-out type; one per surface.
  Every other control is a ruled text control that never competes with it.
- **Selection** — a single 3px ink edge on the row's leading cell only.

## Browser surfaces

Selection, caret, scrollbars, focus rings, underline offset and tabular numerals
are all themed from the palette rather than left at browser defaults.

## Verified

- Impeccable detector: 0 findings on both pages, static and rendered DOM.
- Palette: all five dataviz checks pass against `#0D1012`.
- 393 tests pass.
- Desktop 1440 and mobile 375 inspected; no horizontal page scroll (wide tables
  scroll inside their own container).
