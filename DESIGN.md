---
name: CanopyIntel
description: Editorial Dark Forest design system for subnational forest loss intelligence
colors:
  bg-base: "#0B1410"
  surface-raised: "#121E18"
  surface-card: "#16241D"
  border-hairline: "#24382D"
  border-subtle: "#1F3127"
  text-primary: "#E9E6DA"
  text-secondary: "#9FB0A4"
  text-muted: "#6B7F73"
  accent-moss: "#7FB069"
  accent-moss-glow: "rgba(127, 176, 105, 0.12)"
  tier-low: "#7FB069"
  tier-medium: "#D9B84A"
  tier-high: "#E07A3F"
  tier-critical: "#C8402F"
typography:
  display:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "2.4rem"
    fontWeight: 500
    lineHeight: 1.15
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "Newsreader, Georgia, serif"
    fontSize: "1.45rem"
    fontWeight: 500
    lineHeight: 1.3
    letterSpacing: "-0.015em"
  body:
    fontFamily: "IBM Plex Sans, -apple-system, BlinkMacSystemFont, sans-serif"
    fontSize: "0.92rem"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: "IBM Plex Sans, sans-serif"
    fontSize: "0.78rem"
    fontWeight: 600
    letterSpacing: "0.04em"
  mono:
    fontFamily: "IBM Plex Mono, monospace"
    fontSize: "0.85rem"
    fontWeight: 500
rounded:
  sm: "4px"
  md: "8px"
  lg: "10px"
  xl: "12px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  card:
    backgroundColor: "{colors.surface-card}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.xl}"
    padding: "1.1rem 1.25rem"
  kpi-card:
    backgroundColor: "{colors.surface-card}"
    textColor: "{colors.text-primary}"
    rounded: "{rounded.lg}"
    padding: "0.85rem 1.1rem"
  status-pill:
    backgroundColor: "{colors.surface-card}"
    textColor: "{colors.accent-moss}"
    rounded: "{rounded.full}"
    padding: "0.25rem 0.75rem"
---

# CanopyIntel Design System (DESIGN.md)

## Overview
CanopyIntel employs an **Editorial Dark Forest** aesthetic designed for environmental researchers, conservation policymakers, and evaluators. The interface balances high-density analytical telemetry with the tactile calm of a field monograph.

- **Mode:** Operate & Experience (High-density analytical dashboards with editorial clarity)
- **Palette Identity:** Deep dark forest canopy tones (`#0B1410` base, `#121E18` surfaces, `#7FB069` moss accents).
- **Zero Anti-Patterns:** No generic neon gradients, no purple-blue AI tropes, no uncalibrated saturated rainbow palettes.

## Colors
- **Canvas Base (`#0B1410`):** Extremely deep, desaturated forest green canvas providing high-contrast readability without eye fatigue.
- **Raised Surfaces (`#121E18`, `#16241D`):** 1-tier and 2-tier elevated card planes with clean hairline borders (`#24382D`).
- **Typography Tones:**
  - `Primary (#E9E6DA)`: Warm antique bone for maximum legibility and reduced glare.
  - `Secondary (#9FB0A4)`: Lichen grey for captions, axis labels, and metadata.
  - `Muted (#6B7F73)`: Subdued green-slate for sparklines and borders.
- **Risk Color Ramp:**
  - `Low`: `#7FB069` (Moss green)
  - `Medium`: `#D9B84A` (Ochre gold)
  - `High`: `#E07A3F` (Burnt amber)
  - `Critical`: `#C8402F` (Crimson hazard)

## Typography
- **Headlines & Narrative Context:** *Newsreader* (Google Fonts optical-sized serif). Provides editorial credibility and authoritative presentation.
- **UI Elements & Explanations:** *IBM Plex Sans*. Clean, neutral grotesque font ensuring effortless scanability across complex data grids.
- **Data, Metrics & Tables:** *IBM Plex Mono*. Enforces strict `font-variant-numeric: tabular-nums` to ensure numerical alignment across time series and regression leaderboards.

## Layout
- **Container Scale:** Max width capped at `1400px` for optimal reading scan lines.
- **Gutter System:** Consistent 24px column gutters and 16px component margins.
- **Top Bar:** Fixed persistent brand header with live model version badge, seed status, and evaluation horizon.

## Elevation & Depth
- **Depth Paradigm:** Flat architectural layering using 1px hairline borders (`#24382D`) rather than muddy drop-shadows.
- **Interactive Focus:** Focus outlines use `1px solid #7FB069` with `2px` offset.

## Shapes
- **Card Corners:** `10px` to `12px` rounded radii (`rounded.lg` to `rounded.xl`).
- **Badges & Pills:** Fully pill-shaped (`9999px`) with 1px border.

## Components
- **Schematic 10x5 Tile Cartogram:** Visual matrix of 50 subnational regions colored by composite risk score.
- **KPI Telemetry Cards:** Micro-cards displaying primary metric values in mono numerals, contextual deltas, and inline sparklines.
- **Plotly Editorial Theme:** Transparent backgrounds, `#24382D` zero-lines, and custom hover popups.

## Do's and Don'ts
- **DO** use tabular numerals (`IBM Plex Mono`) for all decimal values and percentages.
- **DO** annotate every chart with a caption explaining the practical conservation implication.
- **DON'T** introduce blue, purple, or neon gradient accents.
- **DON'T** fabricate geographic coordinates or map shapes when data contains abstract regions (`R01`–`R50`).
