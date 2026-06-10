# Direction — panel-grid map of an unstructured (FVCOM) node field

Date: 2026-06-10. Immutable handoff (xfvcom `docs/directions/` convention).
Consumer: `TB-FVCOM/dye/` dye post-processing (`dye_maps.py`, `dye_anim.py`);
reusable by `ersem/` and `hydro/`. Cross-link: the dye-side method/convention
lives in `TB-FVCOM/dye/docs/dye_run_configuration.md` and the campaign doc
`TB-FVCOM/dye/docs/dye_bc_design.md`.

## Why
The dye source-attribution maps (the published Feb-2026 figures) were produced
by a dye-local script `TB-FVCOM/dye/analysis/plot_dye.py`. That rendering is
broadly useful (any unstructured node field as a 2×N panel grid), so it is being
consolidated into `xfvcom` so dye/ersem/hydro all call one renderer and the
dye-local copy can be retired. The user also wants an **OSM real-land** option
(via `xcoast`) for presentation, while keeping the **mesh-boundary land fill as
the default** (it shows exactly where the model has water — important for
reading the computation).

## What
New module `xfvcom/plot/panel_grid.py` exporting `plot_field_panels(...)`:

- Inputs: a list of node-length fields (one per panel), grid `lon/lat/nv`
  (nv 1- or 0-based), optional per-panel `labels` (info-text box) and
  `marker_nodes` (source-node dots), `layout` (panels per row, e.g. `[6, 6]`).
- Style (ported verbatim from `plot_dye.plot_dye_contour`): turbo with
  gray-under / magenta-over, log contour levels (decade-subdivided) or linear,
  per-row colorbars aligned to panel height, info-text box bottom-right, source
  markers, "Longitude/Latitude" axes.
- `land=`:
  - `"mesh"` (DEFAULT): ordered mesh-boundary polygons — white water area,
    lightgray islands, black coastline. Exactly the legacy look.
  - `"osm"`: `xcoast` OSM land overlay (`CoastMask.add_to_plain_mpl`) for a
    nicer real coastline; lazy-imported so xcoast stays optional.
  - `"none"`: no land.
- Returns `{"fig", "axes", "output_path"}`; saves if `output=` given.

## Acceptance
- Reproduces the legacy 2×6 dye map (mesh land) pixel-for-pixel in spirit
  (visual parity with `plot_dye.plot_dye_contour`).
- `land="osm"` renders OSM land when xcoast data is available; degrades to a
  clear error if xcoast/region data is missing.
- `black --check`, `isort --check-only`, `mypy`, `pytest -m "not png"` pass.
- Exported from `xfvcom.plot`.

## Implements
`xfvcom/docs/directions/20260610_panel_grid_map.md`
