# Open-boundary conditions for FVCOM from the JCOPE-T DA full-domain archive

`scripts/run_jcope_obc.py` builds the two files FVCOM reads along an open boundary from the JCOPE-T DA
(JAMSTEC) ocean reanalysis:

| output | content | FVCOM use |
|---|---|---|
| `tb_tsobc_<tag>_<years>.nc` | temperature `obc_temp` and salinity `obc_salinity` on the FVCOM sigma layers at every OBC node, hourly | time-series T/S nudging at the open boundary |
| `tb_julian_obc_<tag>_<years>.nc` | sea-surface height at every OBC node, hourly | elevation open-boundary forcing |

The reanalysis comes from the **jcopetda full-domain archive** (one NetCDF per UTC day,
2018-06-04 .. 2025-12-31, the whole 1190 x 1190 x 46-level JCOPE grid). Its format is described in
`jcopetda/docs/full_domain_format.md`; how it was made and which records are damaged upstream is in
`jcopetda/docs/corrupt_upstream_files.md`.

## What you need

1. **The archive.** Use `$DATA_DIR/jcopetda_provisional`. It is the archive with the two records that the
   data provider cannot replace (salinity 2020-12-15 17Z, zonal velocity 2022-03-12 21Z) filled, so the
   reader needs no special handling. (`$DATA_DIR/jcopetda` is the unfilled original; it declares those
   hours missing and the tool then stops, unless you pass `--missing interpolate`.)
2. **The JCOPE static grid** `basic.nc`: default `$DATA_DIR/jcopetda/grid/basic.nc`. It supplies the real
   JCOPE bathymetry and layer depths, which the vertical interpolation uses.
3. **Your FVCOM mesh files**, the ones the model itself reads:
   `<case>_grd.dat` (node coordinates, UTM), `<case>_dep.dat` (node depths) and `<case>_obc.dat`
   (the open-boundary node IDs, in boundary order).
4. The conda environment `xfvcom` (`conda activate xfvcom`).

## Quick start

```bash
python scripts/run_jcope_obc.py \
    --archive-root "$DATA_DIR/jcopetda_provisional" \
    --fvcom-grid  input/grid/TokyoBay_grd.dat \
    --fvcom-dep   input/grid/TokyoBay_dep.dat \
    --fvcom-obc   input/grid/TokyoBay_obc.dat \
    --years 2020-2025 --n-siglay 30 --utm-zone 54 \
    --tag jcope_v5 --output-dir input/obc/v5
```

A six-year build takes one to one and a half hours on one node, so submit it as a batch job on a supercomputer. A ready
job script for GENKAI is `TB-FVCOM/input/scripts/genkai_create_obc_v5.sh`:

```bash
pjsub -x YEARS=2020-2025 input/scripts/genkai_create_obc_v5.sh          # whole years
pjsub -x YEAR=2022       input/scripts/genkai_create_obc_v5.sh          # one year
pjsub -x YEARS=2020-2025,GRID_TAG=MyBay,DEP=MyBay_dep_cap300.dat input/scripts/genkai_create_obc_v5.sh
```

(`GRID_TAG` is the file-name prefix of the three mesh files, `DEP` the depth file to use.) Outputs go to a
staging directory (`input/goto2023/obc/v5` in TB-FVCOM); existing production files are never overwritten.
Promote a file by copying it and pointing the run's namelist at it after you have checked it.

## Options

| option | meaning |
|---|---|
| `--archive-root DIR` | the full-domain archive (replaces the older `--region-nc/--region-pattern`) |
| `--years 2020-2025`, `--year 2022` | calendar years, each read whole (Jan 1 - Dec 31; 2018 starts on 2018-06-04, the first day of the archive). A list must be increasing and consecutive; `--allow-year-gaps` accepts a skipped year (the time axis then jumps) |
| `--missing error\|interpolate` | what to do with an hour the archive declares missing: stop (default), or interpolate linearly in time. Not needed for `jcopetda_provisional` |
| `--fvcom-grid/--fvcom-dep/--fvcom-obc` | the FVCOM mesh files; `--utm-zone N` (default 54, Tokyo Bay) |
| `--n-siglay N` | uniform sigma layers (default 30) |
| `--sigma-dat FILE` | a sigma-z (SADAPT/GTSZ) `sigma.dat`: the T/S are put on each OBC node's actual layer depths. The mesh and depth files must be the ones the `sigma.dat` was built for |
| `--siglev-npz FILE` | explicit per-node sigma levels; the node IDs and depths stored in the file are checked against your OBC list |
| `--obc-depth-control`, `--obc-depth-control-endpoints observed\|source` | sample at the depths FVCOM substitutes under `OBC_DEPTH_CONTROL_ON = T` (see below) |
| `--tag NAME`, `--output-dir DIR` | output file tag and directory |
| `--no-tsobc`, `--no-elevation` | write only one of the two files |
| `--jobs N` | build years in parallel (more memory) |

### Depth control

With FVCOM's default `OBC_DEPTH_CONTROL_ON = T` the model replaces every OBC node's depth by its inward
neighbour's. The file must then be sampled at the substituted depths, which `--obc-depth-control` computes
from the mesh. For the two end nodes of the boundary string the tool follows a rule measured on a production
run (`observed`: they keep their own depth); `source` follows the Fortran source literally. They differ at
node 3150 (28.9 m vs 56.4 m) on the Tokyo Bay mesh. Compare with the model's output depth `h` after a first
run before trusting a new mesh or rank count (`TB-FVCOM/hydro/docs/obc_depth_consistency.md`).

## What the tool does

* For each OBC node it takes the nearest **ocean** cell of the JCOPE grid (no horizontal interpolation).
* The vertical interpolation uses the real JCOPE depth and layer depths of that cell (`basic.nc`), not an
  assumed uniform sigma, and the FVCOM layer depths `siglay * h` of the node.
* Times are the archive's (hourly; JAMSTEC has not confirmed the time zone, UTC is assumed).
* The output records its source, the missing-record policy, and any filled hours in the file history
  (`ncdump -h`).

## Checks and error messages

The tool stops with a message rather than write a doubtful file. The ones you may meet:

| message (shortened) | meaning / what to do |
|---|---|
| `no data at N requested cell(s) ... land (or fill) in the archive` | the JCOPE grid file calls the cell ocean but the archive has no data there. Move the OBC node or choose another cell |
| `day file(s) missing under ...` | the years requested are outside 2018-06-04 .. 2025-12-31, or the archive is incomplete |
| `hour(s) ... are declared missing` | you pointed at the unfilled archive; use `jcopetda_provisional` or `--missing interpolate` |
| `years ... are not consecutive` | see `--allow-year-gaps` |
| `OBC node ... outside the JCOPE grid`, `must be finite and positive` | wrong coordinates (UTM zone?) or depths in the mesh files |
| `siglev-npz depths differ ...` | the per-node level file belongs to another depth file |

## Validating a result

`scripts/compare_jcope_sources.py` checks, for a few days, that the archive and an older
`jcopetda_region_YYYY.nc` give the same boundary columns (differences are at most half a packing step:
0.005 for temperature and salinity, 0.0005 for velocity and elevation). For Tokyo Bay the new files agree
with the previous region-based build (`v4`, 2020-2022) to 0.005.

## Python API

```python
from xfvcom.io import FullDomainSource, JcopeGrid, JcopeObcGenerator

src  = FullDomainSource("/path/jcopetda_provisional", "2020-01-01", "2020-12-31")
grid = JcopeGrid("/path/jcopetda/grid/basic.nc")
gen  = JcopeObcGenerator(grid=grid, region_nc=src,
                         obc_nodes=ids, obc_lat=lat, obc_lon=lon, obc_h_fvcom=h, n_siglay=30)
gen.write_tsobc("tsobc_2020.nc")
gen.write_elevation("elevation_2020.nc")
```

## Notes on the two filled records

The data provider cannot supply two records that are damaged in the archive (salinity 2020-12-15 17Z, levels
14-18; zonal velocity 2022-03-12 21Z, levels 41-45). `jcopetda_provisional` keeps the good cells of those
records and fills the damaged cells by linear interpolation in time. This is a stand-in, not data: the
interpolation smooths whatever varies faster than two hours (tidal currents). The boundary files contain
temperature, salinity and elevation only, so the damaged velocity record does not enter them; for salinity,
the 13 Tokyo Bay boundary columns of 2020-12-15 17Z are identical in the region-based build and in the new build. Details, cell counts and tolerances: `jcopetda/docs/corrupt_upstream_files.md`.
