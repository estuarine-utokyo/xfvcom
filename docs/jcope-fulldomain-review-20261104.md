# Review log: full-domain JCOPE source for OBC generation (gpt-6.1-sol)

## Round 1 (commit 513cc15 reviewed; fixes in 1419abd)

### Final answer (verbatim)

1. **Major — All-NaN boundary conditions are accepted and written successfully.**  
   Location: [xfvcom/io/jcope_obc_generator.py:449](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:449).

   **Evidence:** The real archive’s TT/ST/EGT masks exclude 4,504 cells marked ocean by `basic.nc`. At `(j, i) = (608, 816)`, `basic.nc` reports ocean and depth `562.57056 m`; the generator selects that cell and returns entirely NaN temperature, salinity, and elevation for 2020-01-01. No exception occurs. An in-memory write also confirmed that NaN forcing is successfully written. Undeclared all-fill hours follow the same path.

   **Fix:** Reject missing OBC columns and nonfinite output values before writing, with variable, time, node, and source-cell diagnostics. Reconcile the grid/archive mask discrepancy before permitting affected cells.

2. **Major — Year lists can produce backward, duplicate, or year-long gaps in forcing time.**  
   Location: [scripts/run_jcope_obc.py:583](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:583).

   **Evidence:** `parse_year_spec` preserves order and duplicates, and concatenation follows that list. Mocked executions of the actual runner returned success for `--years 2021,2020` and `--years 2020,2020`, writing nonmonotonic time axes. The documented `2020,2022` form produces an 8,761-hour transition between complete years, leaving an unrepresented year inside one forcing file.

   **Fix:** Validate and normalize the year list before reading data. Reject duplicates and gaps, or write separate files for separate continuous periods. Validate hourly continuity again before writing.

3. **Minor — Daily time validation checks only the first timestamp and length.**  
   Location: [xfvcom/io/jcope_fulldomain.py:89](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:89).

   **Evidence:** Changing hour 05’s timestamp to hour 04 in an in-memory day file passed construction. The generator then accepted hourly differences containing `1, 0, 2, 1` hours. Interpolation also assumes array positions represent consecutive hours.

   **Fix:** Compare every decoded timestamp against the exact expected 24-hour UTC sequence. Reject fractional, duplicate, missing, or reordered timestamps without truncating them first.

4. **Minor — Spatial and vertical coordinates are not validated across days or against the grid.**  
   Location: [xfvcom/io/jcope_fulldomain.py:94](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:94).

   **Evidence:** A second day with latitude shifted by one degree and reversed level coordinates was accepted and read normally. Only the first day’s horizontal coordinates are retained; level coordinates are never checked. The generator checks the horizontal origin, rather than equality of the complete axes.

   **Fix:** Validate every day’s complete horizontal axes and ordered levels against the expected archive grid, and verify compatibility with `JcopeGrid` before generating forcing.

5. **Minor — Unsupported payload schemas can silently produce incorrect profiles.**  
   Location: [xfvcom/io/jcope_fulldomain.py:155](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:155).

   **Evidence:** An injected TT variable with one vertical level was silently broadcast into all 46 output levels. Separately, a packed value of `1000`, scale `0.01`, and offset `5` decoded as `10` instead of `15`. Dimensions, dtype, fill metadata, offset, and format version are not validated.

   **Fix:** Enforce the supported format contract before reading: dimension names/order/sizes, dtype, packing attributes, fill value, and format version. Reject unsupported schemas rather than allowing broadcasting or incomplete CF decoding.

6. **Minor — `missing_fields` dates are parsed but discarded.**  
   Location: [xfvcom/io/jcope_fulldomain.py:109](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:109).

   **Evidence:** Putting `TT_2019123105` in the 2020-01-01 file produces `_missing == {"TT": [5]}`. Interpolation therefore modifies January 1 at 05Z, regardless of the declared date. Unknown variable names are also retained without validation.

   **Fix:** Strictly validate each declaration’s syntax, variable, uniqueness, and membership in that file’s actual time axis. Derive the index from the complete timestamp.

7. **Minor — Interpolation cannot use available neighbors outside an annual source.**  
   Location: [xfvcom/io/jcope_fulldomain.py:175](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:175).

   **Evidence:** With December 31 at 23Z declared missing and January 1 available, an annual source raises “no valid neighbour on both sides.” Extending the same source through January 1 succeeds. The runner always constructs independent calendar-year sources, so `--years` cannot bridge this gap.

   **Fix:** Read neighboring records outside the requested output period when interpolation needs them, including across year directories, then trim to the requested period.

8. **Minor — Explicit UTC dates are rejected.**  
   Location: [xfvcom/io/jcope_fulldomain.py:64](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:64).

   **Evidence:** `start=end="2020-01-01T00:00:00Z"` fails the first-timestamp check because the requested day is timezone-aware while decoded archive timestamps are naive. This contradicts accepting arbitrary `pandas.Timestamp` inputs and rejects an unambiguous UTC date.

   **Fix:** Define dates as UTC days; convert aware inputs to UTC and remove timezone metadata consistently before normalization and comparison.

9. **Minor — Fractional cell indices are silently truncated.**  
   Location: [xfvcom/io/jcope_fulldomain.py:139](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:139).

   **Evidence:** `read_profile("TT", [0.9], [1.9])` successfully reads cell `(0, 1)`. Mismatched lengths, scalar inputs, and out-of-range indices instead produce incidental indexing exceptions.

   **Fix:** Validate that indices are integral, have matching supported shapes, and lie within bounds before conversion and allocation.

10. **Minor — Invalid vertical geometry can become plausible finite forcing.**  
    Location: [xfvcom/io/jcope_obc_generator.py:145](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:145).

    **Evidence:** Negative FVCOM depth was accepted and produced surface-valued profiles through constant extrapolation. Positive per-node sigma coordinates likewise produced finite surface-valued profiles. NaN depths produced NaN arrays without error. Current validation checks array sizes, but not physical validity.

    **Fix:** Validate finite positive depths and finite, correctly ordered sigma coordinates in `[-1, 0]`, including interfaces, layer placement, and permitted collapsed layers. Apply equivalent validation in the public writers.

11. **Minor — Invalid geographic coordinates silently select distant ocean cells.**  
    Location: [xfvcom/io/jcope_grid.py:124](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:124).

    **Evidence:** On the real grid, `(lat, lon) = (0, 0)` maps to `(17, 117)`. `(NaN, 140)` maps to `(17, 140)` after a NumPy warning. Clipping occurs before the ocean search, so its radius provides no protection against these inputs.

    **Fix:** Reject nonfinite and out-of-domain OBC coordinates before calling the clipping lookup. Preserve clipping only where explicitly intended by the low-level API.

12. **Minor — Malformed OBC records can silently remove nodes or select the last mesh node.**  
    Location: [scripts/run_jcope_obc.py:68](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:68).

    **Evidence:** A mocked three-node OBC file containing IDs `1`, `0`, and `bad` parsed successfully as `[1, 0]`. The malformed row disappears; ID zero becomes mesh index `-1` at line 408. The declared node count is ignored.

    **Fix:** Parse records strictly, verify the declared count, and require unique IDs within `1..n_nodes` before indexing mesh arrays.

13. **Minor — A failed rewrite destroys an existing valid output.**  
    Location: [xfvcom/io/jcope_obc_generator.py:551](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:551), also line 631.

    **Evidence:** Both writers open the destination directly with `"w"`. An in-memory fault injected after three variable creations removed the prior output and left a partial replacement. A malformed coordinate shape also fails after opening and truncating the destination.

    **Fix:** Validate inputs first, write to a temporary sibling file, close successfully, then atomically replace the destination. Remove temporary files on failure.

14. **Minor — Temporal repairs are absent from output provenance.**  
    Location: [xfvcom/io/jcope_fulldomain.py:119](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:119).

    **Evidence:** Sources using `missing="error"` and `missing="interpolate"` have identical names. Output history receives only that name, so repaired variables and timestamps are lost. The provisional archive’s `provisional_fill` metadata is also ignored.

    **Fix:** Propagate the missing-data policy, actual interpolated records, and provisional-fill declarations into output metadata.

15. **Minor — The comparison script can validate the wrong regional cell.**  
    Location: [scripts/compare_jcope_sources.py:53](/home/pj24001722/ku40000343/Github/xfvcom/scripts/compare_jcope_sources.py:53).

    **Evidence:** Unlike the generator, the comparison script does not check regional index bounds. A cell immediately before the region becomes index `-1`, which xarray accepts as the last cell. With homogeneous synthetic fields, the actual comparison function reported all variables `OK` and returned zero despite the requested point being outside the region.

    **Fix:** Check every translated index against the regional dimensions before indexing, and verify coordinate alignment explicitly.

16. **Minor — The comparison script crashes when there are no jointly finite values.**  
    Location: [scripts/compare_jcope_sources.py:70](/home/pj24001722/ku40000343/Github/xfvcom/scripts/compare_jcope_sources.py:70).

    **Evidence:** Comparing all-NaN fields raises `ValueError: zero-size array to reduction operation maximum which has no identity`. This is reachable with the real mask discrepancies described above and prevents a useful validation report.

    **Fix:** Handle an empty finite intersection explicitly, report its mask statistics, and distinguish insufficient comparison coverage from a numerical mismatch.

## Verdict

VERDICT: FAIL (0 blocker, 2 major, 14 minor, 0 nit)

### Prompt

```markdown
# Review request, round 1: full-domain JCOPE-T DA source for open-boundary generation (xfvcom)

Read-only review of the git repository at the current directory. Do NOT modify files. You may run
read-only commands, python in memory, mocks and fault injections (use the python at
/home/pj24001722/ku40000343/mambaforge/envs/xfvcom/bin/python). Answer in English as Markdown.

## Goal
World-class correctness and robustness. Report every defect you can substantiate, of any severity, in
or outside the change, including pre-existing ones.

## What was done
`git show 513cc15` adds `xfvcom/io/jcope_fulldomain.py` (`FullDomainSource`), lets
`xfvcom/io/jcope_obc_generator.py::JcopeObcGenerator` take it instead of a `jcopetda_region_YYYY.nc`
path, and adds `--archive-root` / `--missing` to `scripts/run_jcope_obc.py`. The source reads hourly
columns of the OBC cells from the jcopetda full-domain archive: one NetCDF per UTC day
`<root>/YYYY/jcopetda_YYYYMMDD.nc`, int16 packed (scale_factor 0.01 TT/ST, 0.001 UT/VT/EGT, fill -32768),
24 hourly steps, 46 levels, 1190x1190, chunks [1,46,64,128]; global attribute `missing_fields`
(e.g. `ST_2020121517`) lists hours whose raw record is absent (the hour is then all fill). Format
description: /home/pj24001722/ku40000343/Github/jcopetda/docs/full_domain_format.md. Real data:
`$DATA_DIR/jcopetda_provisional` (= /home/pj24001722/share/Data/jcopetda_provisional; same files as the
archive except two days in which damaged hours were filled by time interpolation, so `missing_fields` is
empty there). The grid is the JCOPE-T DA grid of basic.nc (JcopeGrid); OBC cells are chosen by
`JcopeGrid.find_nearest_ocean` as before, and the horizontal index (j, i) in basic.nc equals the index in
the day file.
Validation already done: synthetic tests (tests/test_jcope_fulldomain.py) and a real-data comparison
against jcopetda_region_2020.nc (scripts/compare_jcope_sources.py, 13 OBC nodes, 72 h, differences within
half a packing step).
Out of scope (deliberate): horizontal interpolation (nearest ocean cell, as the region path); the
profile cache (not used in archive mode); region-file behaviour (unchanged).

## Please
A fresh, unrestricted audit of `FullDomainSource`, the generator and runner changes, and everything they
touch: time axis and UTC handling, chunk grouping and fancy indexing, unpacking/NaN, missing-hour
handling (error / interpolate), year boundaries (Dec 31 -> Jan 1 across files), pickling for joblib
(`--jobs`), memory and speed for 6 years x 13..few hundred nodes, interaction with
`build_tsobc_arrays`/`vertical_interp` when a column is NaN, `--years` ranges outside the archive
(archive covers 2018-06-04 .. 2025-12-31), and anything else that could give a wrong open boundary
condition without an error.

## Severity
- blocker: produces wrong scientific results or loses data in normal use
- major: a failure or wrong result that can be accepted as success, in a realistic path
- minor: needs unusual input or an injected fault, or is a clear robustness/clarity defect
- nit: style, wording, dead code

## Required output
Numbered findings, each with severity, file:line, a reproduction or evidence, and a concrete fix. Then
`## Verdict` with exactly one line: `VERDICT: PASS` (no finding of any severity) or
`VERDICT: FAIL (<n> blocker, <n> major, <n> minor, <n> nit)`.
```

### Triage

| id | severity | verified | correct | action |
|---|---|---|---|---|
| 1 | major | reviewer's reproduction (cell (608,816): ocean in basic.nc, land in the archive); code path read | yes | FullDomainSource raises on a requested cell with no data (test_land_in_the_archive...) |
| 2 | major | reproduced by reading parse_year_spec/concatenation | yes | check_year_list: increasing, unique, consecutive (--allow-year-gaps); tests |
| 3-6 | minor | reproduced by reading the code | yes | exact 24-hour axis, axes/level/packing/scale validation, strict missing_fields; tests |
| 7 | minor | code path | yes | interpolate mode reads the neighbouring day outside the period; test |
| 8 | minor | yes | yes | UTC-aware dates converted; test |
| 9 | minor | yes | yes | integral, in-range, equal-length indices; test |
| 10-11 | minor | yes | yes | finite positive depth, sigma in [-1,0] ordered, finite in-domain OBC coordinates; test |
| 12 | minor | yes | yes | strict OBC file parser, ids unique >= 1 and <= n_nodes; test |
| 13 | minor | yes | yes | writers write a .tmp sibling and os.replace; test |
| 14 | minor | yes | yes | provenance() carries policy, interpolated hours and provisional_fill into history |
| 15-16 | minor | yes | yes | comparison script: region bounds/axes check, empty intersection reported |

## Round 2 (commit 1419abd reviewed; fixes in 894e28c, jcopetda 8ac3a97)

### Final answer (verbatim)

Several round-1 fixes remain partial. The real archive passes the new metadata checks, but the audit found additional correctness and robustness defects.

Validation used the specified Python, in-memory NetCDFs, read-only real-data reads, and filesystem mocks. No files were modified.

- All **2,192 daily files for 2020–2025** passed construction.
- A real **13-node, one-day** build produced finite forcing and matched regional data within packing and float32 rounding tolerances.
- Replaying both damaged-record repairs reproduced the provisional payloads exactly, including the documented replacement counts.
- No substantial runtime regression was established. The new validity check does add peak memory, reported below. A complete six-year payload build with process workers was not run.

**Status of previous findings**

| Round-1 finding | Status | Reason |
|---|---|---|
| 1. All-NaN columns | RESOLVED | The full-domain reader now rejects requested all-NaN columns. The unchanged regional path still needs protection; see finding 3. |
| 2. Year lists | PARTIAL | Lists are checked, but actual source timestamps and concatenated continuity are not. |
| 3. Daily time axis | RESOLVED | Every decoded timestamp is compared against the expected hourly sequence. |
| 4. Spatial/vertical axes | RESOLVED | Cross-day axes, level values, and full-domain/grid horizontal compatibility are checked. Schema holes remain under finding 5’s scope. |
| 5. Unsupported schemas | PARTIAL | Dimension lengths and coordinate dimensions remain unchecked; unsupported decoding attributes and scales are accepted. |
| 6. Missing declarations | RESOLVED | Variable, date, hour membership, syntax, and duplicate declarations are checked. |
| 7. Outside-period neighbors | PARTIAL | The additional days fix isolated boundary gaps, but irrelevant halo gaps can now prevent otherwise valid reads. |
| 8. UTC-aware dates | RESOLVED | Aware inputs are converted consistently to UTC days. |
| 9. Cell indices | PARTIAL | Ordinary fractional and out-of-range indices are rejected; complex indices are silently converted. |
| 10. Vertical geometry | PARTIAL | Range/order checks were added, but endpoints, layer placement, source geometry, and public-writer validation remain incomplete. |
| 11. Geographic coordinates | PARTIAL | Nonfinite and distant coordinates are rejected, but a full extra grid spacing remains accepted outside each edge. |
| 12. OBC parsing | PARTIAL | Zero/duplicate IDs are rejected, but declared counts and malformed initial records remain unchecked. |
| 13. Atomic writers | PARTIAL | Ordinary single-writer failures preserve the destination; shared temporary filenames introduce collision and symlink hazards. |
| 14. Repair provenance | PARTIAL | Policy and declarations are included, but declarations are incorrectly presented as repairs actually performed. |
| 15. Comparison bounds | RESOLVED | Regional negative indexing and regional-axis alignment are checked. Additional comparison gaps appear below. |
| 16. Empty comparison intersection | RESOLVED | Empty intersections now produce an insufficient-coverage report and failure status. |

**Findings**

1. **Blocker — Nearest-ocean lookup can select a more distant cell during normal operation.**  
   Location: [jcope_grid.py:135](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:135).

   **Evidence:** The search stops at the first nonempty square. On the real grid, requesting cell `(737,856)` selects `(745,865)`, squared distance `145`, although ocean cell `(737,866)` has squared distance `100`. A deterministic sample found this error at **179 of 651** land points having ocean within the search window. Distances also use the rounded cell rather than the actual requested position. Both source paths consequently can sample the wrong scientific column.

   **Fix:** Search all candidates within the defined radius, or use a spatial index with a valid stopping bound. Measure distance from the requested position.

2. **Major — Valid year lists still permit backward or duplicated forcing timestamps.**  
   Location: [run_jcope_obc.py:615](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:615).

   **Reproduction:** Run `--years 2020-2021 --region-pattern /path/region.nc`, without `{year}`. Both years read the same file. An execution of the actual runner with in-memory inputs returned `0` and passed times `[58849, 58849.0416667, 58849, 58849.0416667]` to the writer. Consecutive filenames containing wrong years have the same problem.

   **Fix:** Validate actual per-source periods and the concatenated time axis. Reject duplicates, backward steps, and unapproved gaps before publication.

3. **Major — The regional path and public writers still accept nonfinite forcing as success.**  
   Location: [jcope_obc_generator.py:484](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:484), also lines 649 and 705.

   **Reproduction:** An all-NaN regional TT column produces an all-NaN interpolated field without error. The real writer implementation, redirected to an in-memory NetCDF, successfully stored all-NaN temperature; the elevation writer successfully stored infinity. Thus the mask discrepancy fixed for archive mode remains dangerous in regional mode.

   **Fix:** Reject nonfinite final forcing in the builders and both public writers, with time, node, and variable diagnostics. Validate before replacing an existing output.

4. **Major — Profile-cache identity permits scientifically stale profiles.**  
   Location: [run_jcope_obc.py:145](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:145).

   **Evidence:** Cache acceptance checks only node IDs and regional-file byte size. Source path, stored modification time, node coordinates, selected cells, and grid identity are ignored. A mocked cache from another same-sized source was accepted despite a changed modification time. Reusing node IDs after moving nodes or changing the source can mix old profiles with newly calculated geometry.

   **Fix:** Bind caches to a reliable source fingerprint, time/level axes, selected cells, and grid identity. Validate and fully load cache contents inside the error-handling block.

5. **Minor — An output tree aliasing the source destroys archive files.**  
   Location: [make_provisional_tree.py:154](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:154).

   **Reproduction:** With `--root /archive --out /archive`, the ordinary-day branch replaces each source file with a symlink pointing to itself and reports success. A filesystem-mocked execution confirmed this exact operation. An output symlink resolving to the source has the same risk.

   **Fix:** Resolve and reject identical or overlapping source/output trees before any cleanup or mutation; also reject output year directories aliasing source directories.

6. **Minor — One-level payloads can still broadcast into 46-level profiles.**  
   Location: [jcope_fulldomain.py:150](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:150).

   **Reproduction:** Define dimension `level=1`, but define variable `level` on a separate 46-element dimension containing `0..45`. The validator accepts TT with the expected dimension *names*. Reading it successfully returns `(24,46,1)`, repeating the single supplied level throughout the column.

   **Fix:** Check dimension lengths, coordinate dimensions, and complete payload shapes explicitly. Refuse broadcasting during assignment.

7. **Minor — Packing validation still accepts unsupported decoding contracts.**  
   Location: [jcope_fulldomain.py:173](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:173).

   **Reproduction:** In-memory day files with scale `0` and `-0.01` were accepted and decoded packed `1000` as `0` and `-10`. Adding `missing_value=1000` was also accepted, but the reader returned finite `10` because automatic masking is disabled. `_Unsigned` is accepted but ignored.

   **Fix:** Enforce the supported per-variable scales and exact packing contract, rejecting unsupported decoding attributes. Check format-version type/value without truncating it through `int()`.

8. **Minor — Irrelevant halo gaps can break an intact requested period.**  
   Location: [jcope_fulldomain.py:290](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:290).

   **Reproduction:** Request intact January 2 with interpolation enabled. If January 1 exists and declares `TT_2020010100` missing, reading January 2 raises “no valid neighbour on both sides”: the code attempts to repair the unused halo hour. A single halo day also cannot resolve longer edge gaps.

   **Fix:** Repair only requested hours and their necessary dependencies. Extend outward until required valid neighbors are found, without requiring unrelated halo repairs.

9. **Minor — Invalid sigma geometry still produces plausible finite profiles.**  
   Location: [jcope_obc_generator.py:228](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:228).

   **Reproduction:** The generator accepts interfaces `[0,-0.2,-1]` with centers `[-0.9,-0.95]`, interfaces ending at `-0.3`, and entirely zero interfaces/centers. These pass range/order validation and generate finite forcing. Public writers also accept positive sigma coordinates and negative depths.

   **Fix:** Require surface/bed endpoints, centers within their corresponding interfaces, and an explicit policy for collapsed layers. Share geometry validation with the public writers.

10. **Minor — Invalid JCOPE depth geometry is not rejected.**  
    Location: [jcope_obc_generator.py:240](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:240).

    **Reproduction:** Replacing sampled `ZZ` values with NaNs permits construction and produces all-NaN forcing. Sign-invalid finite source depths can instead produce finite constant extrapolation, hiding the error.

    **Fix:** Validate sampled active-level geometry wherever source values are finite, including sign, ordering, and consistency with source depth.

11. **Minor — OBC parsing still silently drops nodes.**  
    Location: [run_jcope_obc.py:79](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:79).

    **Reproduction:** A header declaring three nodes followed by only two valid records returns two IDs successfully. A first record `1 bad 1` is silently skipped, and subsequent valid records are accepted.

    **Fix:** Parse exactly one recognized header, enforce its count, and reject every malformed data record, including the first.

12. **Minor — Depth-file coordinates and row correspondence are discarded.**  
    Location: [run_jcope_obc.py:98](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:98).

    **Evidence:** Only the third column is retained; the header and coordinates are ignored. The runner checks only total row count. Reordering depth records or supplying another same-sized mesh’s depth file therefore assigns depths to the wrong OBC nodes without error.

    **Fix:** Parse the complete declared depth block strictly and compare its ordered coordinates with the mesh before selecting boundary depths.

13. **Minor — Public APIs silently coerce invalid node identifiers.**  
    Location: [jcope_obc_generator.py:146](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:146).

    **Reproduction:** `obc_nodes=[0]` is accepted; `[1.9]` becomes `[1]`. Duplicate identifiers are also unchecked outside the runner.

    **Fix:** Validate nonempty one-dimensional, integral, unique, positive, representable IDs before conversion, in both the generator and writers.

14. **Minor — Geographic validation still permits clipping outside the domain.**  
    Location: [jcope_obc_generator.py:154](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:154).

    **Reproduction:** On a grid beginning at latitude `17` with spacing `1`, latitude `16.25` is accepted and clipped to the edge. This lies outside even a half-cell extension of the grid.

    **Fix:** Define the permitted domain boundary explicitly and enforce it; do not allow an arbitrary extra full grid spacing.

15. **Minor — Complex cell indices are silently converted to real indices.**  
    Location: [jcope_fulldomain.py:240](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:240).

    **Reproduction:** `_indices([0+3j], [0+2j])` returns integer arrays containing zero, emitting only `ComplexWarning`.

    **Fix:** Reject non-real numeric dtypes before testing integrality or converting indices.

16. **Minor — Shared temporary filenames undermine atomic-write safety.**  
    Location: [jcope_obc_generator.py:546](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:546); also [make_provisional_tree.py:158](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:158).

    **Evidence:** Writers share `<destination>.tmp`. Concurrent calls can overwrite or remove one another’s temporary file. An existing temporary symlink can redirect truncation to another file; a filesystem mock confirmed that the unrelated target was overwritten on failure. The tree builder additionally deletes matching temporary files without establishing ownership.

    **Fix:** Create unique temporary siblings exclusively, clean up only owned files, and serialize tree reconciliation where needed.

17. **Minor — The new annual NaN check increases peak memory with `--jobs`.**  
    Location: [jcope_fulldomain.py:304](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:304).

    **Reproduction:** Measuring this expression for `(8760,46,500)` showed **205,861,054 bytes** of additional peak allocation. Six simultaneous worker checks can add approximately **1.24 GB**, beyond the already allocated profiles and outputs.

    **Fix:** Check daily or chunk-sized blocks, preserving missing-hour interpolation semantics, instead of materializing an annual boolean array.

18. **Minor — Provenance claims repairs that were never performed.**  
    Location: [jcope_fulldomain.py:215](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:215).

    **Reproduction:** Request only January 2 TT while a January 1 halo declares missing UT. `provenance()` claims “UT linearly interpolated” at January 1 despite UT never being read and the hour being outside the output. Provenance is computed from declarations before any reads.

    **Fix:** Track actual repairs and output dependencies. Return that provenance from workers and assemble it from worker results. Preserve `provisional_method` when reporting archive repairs.

19. **Minor — The comparison script can still validate misaligned data successfully.**  
    Location: [compare_jcope_sources.py:49](/home/pj24001722/ku40000343/Github/xfvcom/scripts/compare_jcope_sources.py:49), also line 81.

    **Reproduction:** With constant fields, an actual full-domain source whose axes were shifted by five degrees produced five `OK` results and exit `0`. A regional time axis duplicating 00Z and omitting 01Z likewise passed. Source/grid axes and exact selected timestamps are never compared.

    **Fix:** Check both sources’ horizontal alignment and exact time-index equality before comparing values; explicitly compare vertical coordinates too.

20. **Minor — Reference-level selection contradicts “two on each side.”**  
    Location: [make_provisional_tree.py:80](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:80).

    **Evidence:** The code selects the four closest intact levels overall. For damaged levels `4..8`, level 4 can use references `[3,2,1,0]`, rather than `[3,2,9,10]`. An injected scatter pattern changed the tolerance from the documented rule’s result and retained a moderate outlier.

    **Fix:** Select up to two intact references on each side with a documented boundary fallback, or change the documented algorithm and justify the asymmetric selection.

21. **Minor — Moderate-only corruption bypasses the tolerance test entirely.**  
    Location: [make_provisional_tree.py:67](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:67).

    **Reproduction:** A UT level containing one injected `9 m/s` value among `1 m/s` values, with interpolated values of `1 m/s`, reports zero garbage replacements and retains `9`. Without extreme values, the level is automatically classified as intact and becomes a potential reference.

    **Fix:** Use independently established damaged-level information or additional integrity checks. Absence of extreme values alone cannot establish intactness.

22. **Minor — A wholly corrupt level crashes during diagnostic printing.**  
    Location: [make_provisional_tree.py:95](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:95).

    **Reproduction:** With two intact levels and one entirely corrupt sea level, replacement values are computed, but `d.max()` raises a zero-size reduction error. The preceding percentile already handles this case.

    **Fix:** Handle empty retained-cell diagnostics consistently, including mean and maximum.

23. **Minor — Damaged EGT records are unsupported despite the general archive interface.**  
    Location: [make_provisional_tree.py:65](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:65).

    **Reproduction:** Calling the damaged-record path for EGT raises `KeyError: 'EGT'`; the bounds/tolerance dictionaries and level-oriented implementation cover only 3-D variables. This path is reached when a declared EGT gap has a raw record in `--damaged-dir`.

    **Fix:** Implement appropriate 2-D repair handling, or explicitly route unsupported variables to declared pure interpolation with clear provenance.

24. **Minor — Rebuilding the provisional tree leaves obsolete filled days behind.**  
    Location: [make_provisional_tree.py:173](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:173).

    **Reproduction:** Remove a source day that previously required a filled copy. Rebuilding deletes obsolete symlinks only, leaving the regular filled file available to readers. Entire removed source-year directories are not reconciled either. A filesystem-mocked run confirmed the retained stale copy and successful exit.

    **Fix:** Reconcile an owned output inventory against the complete source inventory, including filled copies and removed years.

25. **Minor — Explicit paths unnecessarily require `DATA_DIR`.**  
    Location: [make_provisional_tree.py:153](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:153).

    **Reproduction:** With `DATA_DIR` unset, supplying both `--root` and `--out` still raises `KeyError: 'DATA_DIR'`.

    **Fix:** Read and require the environment variable only for defaults that are actually needed.

26. **Minor — The document asserts an unsupported error bound against truth.**  
    Location: [corrupt_upstream_files.md:248](/home/pj24001722/ku40000343/Github/jcopetda/docs/corrupt_upstream_files.md:248).

    **Evidence:** The code bounds deviation from the quantized temporal interpolant, while the document says retained garbage has error below tolerance relative to truth. The interpolant’s unknown error must also be included; the stated scientific bound does not follow from the algorithm.

    **Fix:** Describe the bound against the interpolant and explicitly retain uncertainty about truth. The replacement counts and payloads themselves matched the run log and replay.

27. **Minor — The atomic-write regression test never exercises a write failure.**  
    Location: [test_jcope_fulldomain.py:306](/home/pj24001722/ku40000343/Github/xfvcom/tests/test_jcope_fulldomain.py:306).

    **Reproduction:** Its invalid field shape raises before `nc.Dataset` is called. Running that stimulus against the undecorated writer made zero Dataset calls, so the test would pass with the old destructive implementation.

    **Fix:** Supply valid inputs and inject failure during variable creation, payload writing, or close. Check destination preservation and temporary-file cleanup for both writers.

28. **Nit — Temperature units are misspelled.**  
    Location: [jcope_obc_generator.py:64](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:64).

    **Evidence:** Every temperature output receives `units="Celcius"`, and the schema test enshrines that spelling.

    **Fix:** Use a correctly spelled temperature unit such as `degrees_Celsius` and update the assertion.

## Verdict
VERDICT: FAIL (1 blocker, 3 major, 23 minor, 1 nit)

### Prompt

```markdown
# Review request, round 2: full-domain JCOPE-T DA source for open-boundary generation (xfvcom)

Read-only review of the git repository at the current directory. Do NOT modify files. You may run
read-only commands, python in memory (/home/pj24001722/ku40000343/mambaforge/envs/xfvcom/bin/python),
mocks and fault injections. Answer in English as Markdown.

## Goal
World-class correctness and robustness. Report every defect you can substantiate, of any severity, in or
outside the change, including pre-existing ones.

## What was done
Round 1 reviewed commit 513cc15 (`FullDomainSource` in `xfvcom/io/jcope_fulldomain.py`, the generator's
full-domain path, `scripts/run_jcope_obc.py --archive-root`). All 16 findings were accepted and fixed in
commit 1419abd (`git show 1419abd`); the log with the round-1 answer and our triage is
`docs/jcope-fulldomain-review-20261104.md`. Data and format as in round 1: archive root
`$DATA_DIR/jcopetda_provisional` (= /home/pj24001722/share/Data/jcopetda_provisional), format description
/home/pj24001722/ku40000343/Github/jcopetda/docs/full_domain_format.md. Per finding:
1 (all-NaN columns accepted): `FullDomainSource._read` now raises when a requested cell has no data at some
hour (a column that is NaN at every level); the cell is NOT silently moved. 2 (year lists): `check_year_list`
in the runner (increasing, unique, consecutive unless `--allow-year-gaps`). 3-6, 8, 9: exact 24-hour UTC axis,
axes/levels/packing/scale validation per day, strict `missing_fields` parsing, UTC-aware dates, integral
in-range indices. 7: in "interpolate" mode the day before/after the period is read when it exists.
10-12: validation of depth/sigma/coordinates in the generator, strict OBC node file parsing. 13: atomic
writers (`_atomic_write`). 14: `provenance()` goes into the output history. 15-16: comparison script.
Real data not yet exercised by the new strict code: a full 2020-2025 build (job running at review time).

Also in scope as a second, smaller item: the tool that makes the archive tree the OBC builder reads,
`/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py` (commit b960c0e there) and its
documentation section "2026-11-04: the two records that stay damaged" in
`docs/corrupt_upstream_files.md`; an independent reviewer already found that a global tolerance kept
garbage cells (fixed with a per-level tolerance). Please check the per-level tolerance logic and the claims
in the document against the code (the log of the last run: logs/make_provisional_tree.7252645.log in that
repository).

## Previous rounds
Round 1: 16 findings (2 major, 14 minor), all fixed; see the log file named above for the finding-by-finding
triage. None was rebutted.

## Please
1. Status of every previous finding: RESOLVED / PARTIAL / NOT RESOLVED, with reasons.
2. Defects introduced by the fixes (regressions), including performance regressions for a 6-year build
   with --jobs, and behaviour of the unchanged region-file path.
3. A fresh, unrestricted audit of the whole scope and everything it touches (FullDomainSource, generator,
   runner, writers, tests, make_provisional_tree.py).

## Severity
- blocker: produces wrong scientific results or loses data in normal use
- major: a failure or wrong result that can be accepted as success, in a realistic path
- minor: needs unusual input or an injected fault, or is a clear robustness/clarity defect
- nit: style, wording, dead code

## Required output
Numbered findings, each with severity, file:line, a reproduction or evidence, and a concrete fix. Then
`## Verdict` with exactly one line: `VERDICT: PASS` (no finding of any severity) or
`VERDICT: FAIL (<n> blocker, <n> major, <n> minor, <n> nit)`.
```

### Triage

| id | severity | correct | action |
|---|---|---|---|
| 1 | blocker | yes (pre-existing; reproduced by the reviewer on the real grid, 179/651 land points) | find_nearest_ocean searches all candidates, Euclidean distance from the requested position; the 13 TB-FVCOM nodes select identical cells before/after |
| 2 | major | yes | per-source period / hourly-step validation of the concatenated axis; same-file patterns refused |
| 3 | major | yes | finite-forcing validation in builders and both writers |
| 4 | major | yes (pre-existing) | cache bound to source/grid fingerprint, cells, coordinates |
| 5 | minor | yes | jcopetda: root/out must be separate trees |
| 6-19 | minor | yes | xfvcom 894e28c, each with a regression test |
| 20-25 | minor | yes | jcopetda 8ac3a97: two reference levels per side, --damaged-levels (moderate-only corruption), 2-D variables, stale-copy reconciliation, DATA_DIR only when needed |
| 26 | minor | yes | document states the bound against the interpolation |
| 27 | minor | yes | failure injection test for both writers |
| 28 | nit | NO (rebutted) | 'Celcius' is the convention of the existing production OBC files (the module docstring: following the conventions already in use); FVCOM does not parse the string; changing it would make new and old files differ |

## Round 3 (commit 894e28c reviewed; fixes in 48b69af, jcopetda 0624221)

### Final answer (verbatim)

Reviewed xfvcom `894e28c` and the jcopetda tool at `8ac3a97`, including the updated documentation at `250406a`. No files were modified.

Validation included 27 existing test cases executed through a read-only harness, diskless NetCDF fault injections, all **2,192 daily files for 2020–2025**, and real production-node reads. The real one-day comparison passed for all five variables. Six simultaneous processes calling `_process_year`, each using one production day, returned finite forcing.

No substantial performance regression was established. Annual source objects serialized to approximately 175 KB, and the day-sized validity check removes round-2 #17’s annual boolean allocation. A complete six-year payload build and actual joblib `--jobs` orchestration were **not** run.

**Round-2 status**

| # | Status | Assessment |
|---|---|---|
| 1 | RESOLVED | Searches all ocean candidates and minimizes distance from the requested position. A tie-documentation discrepancy remains under finding 20. |
| 2 | RESOLVED | Reused source paths, wrong source years, duplicate times, and unapproved discontinuities are rejected. Finding 18 concerns additional helper validation. |
| 3 | PARTIAL | Ordinary NaN/Inf forcing is rejected; masked arrays and conversion overflow bypass protection. Findings 3–4. |
| 4 | RESOLVED | Cache acceptance checks the source/grid fingerprint, nodes, coordinates, cells, times, and profile shapes. |
| 5 | PARTIAL | Root-level aliases are rejected; output year-directory aliases remain destructive. Finding 5. |
| 6 | RESOLVED | Dimension lengths, coordinate dimensions, and payload shapes are checked. Finding 15 identifies a separate coordinate-mask loophole. |
| 7 | RESOLVED | Supported packing scales, decoding attributes, offsets, and integer format version are enforced. |
| 8 | PARTIAL | Irrelevant halo hours are excluded and longer gaps work; unused variables can still force unnecessary halo dependencies. Finding 7. |
| 9 | RESOLVED | Shared validation checks sigma endpoints, ordering, layer placement, and permitted collapsed layers. |
| 10 | RESOLVED | Sampled source depths and active-level geometry are validated. |
| 11 | RESOLVED | Declared counts and malformed initial records are checked. |
| 12 | RESOLVED | Depth-file coordinates and row correspondence are checked against the mesh. |
| 13 | PARTIAL | Float32 identifiers can overflow the supposedly validated int32 result. Finding 13. |
| 14 | RESOLVED | The permitted domain is explicitly limited to the grid plus half a spacing. |
| 15 | RESOLVED | Complex and boolean cell indices are rejected. |
| 16 | PARTIAL | xfvcom uses exclusive unique temporaries; the tree builder still deletes other runs’ active temporaries. Finding 6. |
| 17 | RESOLVED | Archive validity checks operate on daily blocks. |
| 18 | PARTIAL | Actual repairs and worker provenance are collected; repaired halo dependencies are omitted. Finding 8. |
| 19 | PARTIAL | Source/grid, vertical, and time alignment checks exist; regional horizontal tolerance remains too loose. Finding 16. |
| 20 | RESOLVED | Reference selection uses up to two intact levels per side, with the documented boundary fallback. |
| 21 | RESOLVED | Valid explicit damaged-level declarations handle moderate-only corruption. Declaration validation and diagnostics have defects under findings 9–10. |
| 22 | RESOLVED | Empty retained-cell diagnostics no longer crash. Finding 11 concerns empty reference levels. |
| 23 | RESOLVED | EGT explicitly falls back to pure temporal interpolation. |
| 24 | RESOLVED | Removed days, filled copies, and removed years are reconciled. |
| 25 | RESOLVED | Explicit paths work without `DATA_DIR`. |
| 26 | RESOLVED | The document correctly bounds deviation from interpolation, without asserting a bound against truth. |
| 27 | RESOLVED | Both writer tests now inject failure during payload-variable creation. |
| 28 | WITHDRAWN | Rebuttal accepted. A production OBC file inspected during this review uses `Celcius`; retaining that established output convention is reasonable. |

**Findings**

1. **Minor — The regional generator does not validate complete horizontal axes.**  
   Location: [jcope_obc_generator.py:326](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:326).

   **Evidence:** Doubling a synthetic region’s longitude spacing while preserving its origin passes construction and produces forcing. For requested longitude `117.1666667`, the reader samples a regional column labelled `117.2222222`. Only the origin and translated index bounds are checked.

   **Fix:** Require the complete regional latitude/longitude axes to match the corresponding contiguous grid block, with explicit absolute tolerances.

2. **Minor — Regional vertical coordinates and dimension order can silently change profile interpretation.**  
   Location: [jcope_obc_generator.py:363](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:363).

   **Evidence:** Reversing both the regional level coordinate and its payload preserves the physical dataset, but the generator ignores the coordinate and assigns values to ascending JCOPE indices. The reproduced temperature difference was **8.893334°C**, with entirely finite output. The reader also assigns remaining dimensions positionally without transposing them.

   **Fix:** Validate the supported level coordinate and dimension schema, and explicitly transpose accepted variables into canonical order before reading.

3. **Major — Masked arrays bypass missing-forcing protection.**  
   Locations: [jcope_obc_generator.py:761](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:761), [jcope_obc_generator.py:878](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:878), and [jcope_obc_generator.py:497](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:497).

   **Evidence:** Both writers successfully wrote fully masked inputs whose underlying values were finite; the resulting NetCDF variables contained missing values. Supplying fully masked raw profiles instead discarded the mask and produced finite temperature `17` everywhere. Masked arrays are ordinary outputs of netCDF4 reads.

   **Fix:** Preserve and inspect masks before conversion. Reject masked forcing, or convert masks to NaN before validation and interpolation.

4. **Minor — Finite float64 inputs can become infinite written forcing.**  
   Locations: [jcope_obc_generator.py:761](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:761) and [jcope_obc_generator.py:878](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:878).

   **Evidence:** Passing `1e300` to either writer passes finite validation. Conversion into the `f4` variable emits a warning, stores infinity, and returns success.

   **Fix:** Convert to the output dtype under checked conversion, validate that converted array, and write that same array.

5. **Minor — Output year-directory aliases still permit source-file destruction.**  
   Location: [make_provisional_tree.py:228](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:228).

   **Evidence:** With separate roots `/archive` and `/output`, let `/output/2020` symlink to `/archive/2020`. The root guard passes. A filesystem-mocked execution reached an `os.replace` that would replace the source day with a symlink pointing to itself, then reported success.

   **Fix:** Reject aliased output subdirectories before cleanup or publication. Verify that every output parent resolves inside the output tree and outside the source tree.

6. **Minor — Tree startup cleanup deletes other runs’ active temporary files.**  
   Location: [make_provisional_tree.py:208](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:208).

   **Evidence:** Every invocation unconditionally unlinks all `.provtmp-*` entries. A mocked second invocation deleted another run’s active temporary. That run subsequently cannot rename its completed file. UUID filenames do not protect against this cleanup.

   **Fix:** Hold an exclusive reconciliation lock before cleanup and rebuilding, or establish ownership and liveness before removing a temporary.

7. **Minor — An unused variable’s gap can prevent otherwise valid reads.**  
   Location: [jcope_fulldomain.py:175](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:175).

   **Evidence:** An intact January 2 TT dataset with `UT_2020010200` declared missing cannot be constructed in interpolation mode when January 1 is absent. Construction raises for UT before a TT-only read can occur. The same issue affects elevation-only or T/S-only builds.

   **Fix:** Resolve halo dependencies when each variable is read, or let construction restrict its required variables.

8. **Minor — Provenance omits provisional repairs used through halo dependencies.**  
   Location: [jcope_fulldomain.py:155](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:155).

   **Evidence:** Repairing January 2 00Z TT using a January 1 23Z neighbor marked with `provisional_fill` reports the January 2 interpolation but omits the neighbor’s provisional repair and method. Notes are restricted to requested days, although output depends on the halo value.

   **Fix:** Track the actual interpolation endpoints and include relevant provisional provenance from those dependencies.

9. **Minor — Invalid damaged-level declarations silently disable intended corruption handling.**  
   Location: [make_provisional_tree.py:168](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:168).

   **Evidence:** `UT:3-2` produces an empty set; `UX:2` accepts an unknown variable; repeated variable specifications overwrite earlier ranges. In a moderate-corruption probe, valid `UT:2` replaced `9 m/s` with `1 m/s`, whereas reversed-range and misspelled declarations retained `9 m/s` successfully.

   **Fix:** Validate variable names, ascending ranges, active-level bounds, nonempty selections, and repeated specifications before modifying the tree.

10. **Minor — Explicit damaged levels falsify extreme/moderate replacement counts.**  
    Location: [make_provisional_tree.py:83](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:83).

    **Evidence:** Declaring an otherwise intact level forces its extreme count to one. The reproduced diagnostic was: `1 extreme + -1 moderate garbage cells replaced`, while the actual replacement count was zero. Moderate-only corruption is likewise misclassified.

    **Fix:** Keep the measured extreme count separate from a boolean indicating that a level requires tolerance testing.

11. **Minor — A reference level with no ocean cells crashes repair.**  
    Location: [make_provisional_tree.py:88](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:88).

    **Evidence:** Setting one reference level’s sea mask entirely false makes `_dev` return an empty array. `np.percentile(d, 99.99)` raises `IndexError`. The later empty retained-cell handling does not cover this earlier calculation.

    **Fix:** Exclude empty levels from usable references and require sufficient finite reference scatter, with a clear error or documented fallback.

12. **Minor — Grid origin/spacing attributes are not checked against grid axes.**  
    Location: [jcope_grid.py:65](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:65).

    **Evidence:** Shifting only `grid_origin_lon` by one cell was accepted. The full-domain payload index remained `i=6`, while the geometry lookup moved from depth **300 m to 350 m**. Consequently, valid data were paired with another cell’s vertical geometry.

    **Fix:** Validate finite positive spacing and agreement between the full axes and `origin + index × spacing` during grid construction.

13. **Minor — Float32 node IDs can overflow after passing validation.**  
    Location: [jcope_obc_generator.py:577](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:577).

    **Evidence:** `_validate_obc_nodes(np.array([2**31], dtype=np.float32))` returns `[-2147483648]`, with only a warning. The comparison rounds the int32 maximum to float32 before checking it.

    **Fix:** Perform range checking in a representation that preserves the bound, and verify the converted result before returning it.

14. **Minor — Raw-profile overrides can silently drop times or nodes.**  
    Location: [jcope_obc_generator.py:496](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:496).

    **Evidence:** For a 24-hour source, overrides shaped `(1,46,1)` return successful one-hour forcing. For a two-node generator, overrides shaped `(24,46,1)` return one-node forcing. The public builder succeeds; writers detect the mismatch only later.

    **Fix:** Require every supplied profile to have exactly `(len(self.mjd),46,self.n_obc)` before interpolation.

15. **Minor — Coordinate masks are discarded during archive validation.**  
    Location: [jcope_fulldomain.py:244](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:244).

    **Evidence:** Giving the level coordinate `missing_value=10` makes its tenth entry masked in netCDF4. Construction nevertheless succeeds because `np.asarray` exposes the underlying `10`, allowing comparison with `0..45` to pass.

    **Fix:** Reject masked coordinate entries before conversion, then validate finite coordinate values and supported decoding metadata.

16. **Minor — The comparison script accepts materially shifted regional axes.**  
    Location: [compare_jcope_sources.py:95](/home/pj24001722/ku40000343/Github/xfvcom/scripts/compare_jcope_sources.py:95).

    **Evidence:** Shifting regional longitude by `0.001°` and latitude by `0.0001°` still produced five `OK` results and exit `0` with horizontally homogeneous synthetic fields. Default `np.allclose` relative tolerance permits these shifts, unlike the explicit archive/grid tolerance.

    **Fix:** Use `rtol=0` and a justified small absolute tolerance consistently for both sources.

17. **Minor — Public ocean lookup silently accepts invalid coordinate arrays.**  
    Location: [jcope_grid.py:122](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:122).

    **Evidence:** On an all-ocean grid, latitude `[NaN]` returns a valid edge cell with only a warning. Latitude `[2]` and longitude `[1,3]` return one result, silently ignoring the second longitude. Generator-side validation does not protect direct grid callers.

    **Fix:** Validate finite coordinates and supported, matching shapes before lookup; validate the search radius explicitly as well.

18. **Minor — The new source-time helper accepts NaN timestamps.**  
    Location: [run_jcope_obc.py:175](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:175).

    **Evidence:** `check_time_axis([2020], {2020: np.array([np.nan,np.nan])})` returns successfully because all comparisons with NaN are false. Public writers subsequently reject such times, but this helper does not establish its advertised validation.

    **Fix:** Require finite, nonempty, one-dimensional time arrays before period and step checks.

19. **Minor — Mesh node zero becomes a valid negative array index.**  
    Locations: [dat_reader.py:70](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/dat_reader.py:70) and [grid_obj.py:148](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:148).

    **Evidence:** A three-node mesh containing element connectivity `0 2 3` passes `FvcomGrid.from_dat`, returning connectivity `[-1,1,2]`. Subsequent indexing silently uses the last node for `-1`; this connectivity also reaches the OBC depth-control path.

    **Fix:** Validate connectivity against both bounds before indexing, preferably rejecting nonpositive identifiers while parsing.

20. **Nit — Nearest-ocean tie behavior contradicts its documentation.**  
    Location: [jcope_grid.py:135](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:135).

    **Evidence:** The docstring promises the first cell in `(j,i)` order for ties. On a unit-spaced all-ocean grid, querying `(1.5,1.5)` returns `(2,2)` through the rounded-cell shortcut, although `(1,1)` is the first equally distant cell.

    **Fix:** Apply the stated tie rule consistently, or document the shortcut’s rounding rule.

## Verdict

VERDICT: FAIL (0 blocker, 1 major, 18 minor, 1 nit)

### Triage

Round-2 #28 (the unit string 'Celcius') was WITHDRAWN by the reviewer after our rebuttal. All 20 round-3 findings were accepted: #3 (major, masked arrays bypassed the finite check) and #1,2,4,7,8,12-18,19,20 in xfvcom 48b69af, each with a regression test (31 of the 35 new tests failed on the old code); #5,6,9,10,11 in jcopetda 0624221 (reconciliation lock, no write through linked year dirs, strict --damaged-levels, measured vs declared damage counts, empty reference levels).

## Round 4 (commit 8ec0a6d reviewed; fixes in 56b5870, jcopetda 6698843)

### Final answer (verbatim)

Reviewed xfvcom `8ec0a6d` and jcopetda `0624221`. No files were modified.

Validation included diskless NetCDF reproductions, focused existing regression cases through a read-only harness, construction of all **2,192 daily sources for 2020–2025**, and real production-node reads. The one-day comparison passed for all five variables. Six concurrent process workers, each processing one production day, returned finite forcing. A complete six-year payload build and the actual joblib launch were not run.

**Round-3 findings**

| # | Status | Reason |
|---|---|---|
| 1 | RESOLVED | Complete regional axes are checked with explicit absolute tolerance; altered spacing is rejected. |
| 2 | RESOLVED | Reversed levels are rejected; transposed payloads produce bit-identical forcing. |
| 3 | RESOLVED | Masks become NaN before interpolation/conversion; masked forcing is rejected by both writers. |
| 4 | RESOLVED | Writers validate the converted float32 arrays and reject conversion overflow. |
| 5 | RESOLVED | Linked output year directories are rejected before publication; verified with filesystem mocks. |
| 6 | RESOLVED | Lock acquisition precedes temporary cleanup; a rejected second invocation performs no cleanup. |
| 7 | RESOLVED | Halo dependencies are resolved when the particular variable is read. |
| 8 | RESOLVED | Provisional notes from halo interpolation endpoints are propagated and filtered by the dependent variable. |
| 9 | PARTIAL | Unknown variables, reversed ranges, duplicates and out-of-range levels are rejected. Incomplete ranges remain accepted; finding 8 below. |
| 10 | RESOLVED | Measured extreme counts are separate from declared damage; zero-replacement diagnostics remain nonnegative. |
| 11 | RESOLVED | Empty reference levels are excluded; insufficient usable references produce a clear error. |
| 12 | RESOLVED | Origins, spacings and complete axes are checked; the real `basic.nc` loads successfully. |
| 13 | RESOLVED | Float32/int64 values above the int32 limit are rejected without overflowing conversion. |
| 14 | RESOLVED | Raw overrides must have exactly `(n_time,46,n_obc)`. |
| 15 | RESOLVED | Masked and nonfinite archive coordinate entries are rejected. |
| 16 | RESOLVED | Regional alignment uses `rtol=0` and an explicit absolute tolerance. |
| 17 | RESOLVED | Nonfinite, masked and mismatched coordinates, and invalid radii, are rejected. |
| 18 | PARTIAL | Ordinary NaN, empty, multidimensional and masked times are rejected. Lossy conversion still permits nonfinite complex inputs; finding 9 below. |
| 19 | RESOLVED | The reported out-of-range connectivity identifiers are rejected before indexing. Separate node-record and connectivity-interpretation defects remain below. |
| 20 | RESOLVED | Removing the shortcut applies the documented tie rule consistently; scalar and vector tie tests pass. |

**Findings**

1. **Minor — Explicit damaged-level declarations disable automatic damage detection elsewhere.**  
   Location: [jcopetda/tools/make_provisional_tree.py:85](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:85).

   **Evidence:** `needs_tol` uses the declaration *instead of* measured extreme damage whenever declarations exist. In an eight-level UT probe with a `1 m/s` midpoint, level 2 contained one extreme value and one `9 m/s` garbage value. Automatic detection replaced both. Passing `damaged_levels={5}` retained `9 m/s` successfully on level 2. That damaged level can also contaminate reference scatter.

   This contradicts the documentation’s “extreme values, **or** … named with `--damaged-levels`” rule. The current production declarations cover the known damaged levels, so this reproduction does not establish an error in the existing provisional copies.

   **Fix:** Use `extreme[k] > 0 or k in declared_levels`.

2. **Minor — The mask/overflow fix substantially increases production writer memory.**  
   Locations: [jcope_obc_generator.py:731](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:731), [jcope_obc_generator.py:744](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:744).

   **Evidence:** Every ordinary float32 field is copied to float64 and then copied back to float32. A measured 24 MB input required approximately **72 MB of additional peak allocation**, and the result shared no memory with its input.

   For six years × 300 nodes × 30 layers, each T/S field occupies approximately 1.894 GB. With annual results and concatenated fields retained, conversion raises the T/S buffer peak from approximately **7.6 GB to 15.2 GB**, excluding workers and other allocations. This affects both serial and `--jobs` builds. Cached raw-profile overrides incur the same conversion pattern inside workers.

   **Fix:** Preserve already suitable, unmasked float32 arrays; inspect masks before conversion and perform checked conversion only when needed. Keep finite checks blockwise.

3. **Minor — Mesh node-record identifiers are ignored.**  
   Location: [dat_reader.py:88](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/dat_reader.py:88).

   **Evidence:** A three-node file with node identifiers `0,2,3` is accepted. Records ordered `2,1,3` are also accepted, but coordinates are assigned by row position: the reproduced `x[node 1]` became `401000` instead of `400000`.

   The new connectivity bounds do not detect this. If the depth rows follow the same reordered coordinates, the runner’s depth-coordinate check also passes while OBC identifiers select the wrong coordinates.

   **Fix:** Validate node identifiers for range, uniqueness and required row order, or populate coordinate arrays explicitly by validated identifier.

4. **Minor — `from_dataset` can silently change explicitly zero-based connectivity.**  
   Location: [grid_obj.py:111](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:111).

   **Evidence:** For five nodes, connectivity `[[1],[2],[3]]` with `nv.attrs["start_index"]=0` becomes `[[0],[1],[2]]`. The method ignores the explicit attribute and assumes one-based indexing whenever the minimum is one. This changes geometry for datasets retaining unused nodes, including subsets.

   **Fix:** Honor `start_index`, provide an explicit indexing argument for ambiguous inputs, and annotate exported connectivity with `start_index=0`.

5. **Minor — A two-dimensional depth array can produce finite, incorrectly interpolated profiles.**  
   Location: [jcope_obc_generator.py:173](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:173).

   **Evidence:** Constructor validation checks depth `.size`, not shape. Using the existing three-node synthetic fixture with three sigma layers, replacing depths `[100,200,60]` with shape `(1,3)` succeeds and returns the expected output shape, but temperature differs by up to **5.029335°C**. Broadcasting mixes the node and layer axes. With other layer counts, it instead fails later with an incidental broadcasting error.

   **Fix:** Require `obc_h_fvcom.shape == (n_obc,)` before interpolation; enforce equivalent shapes for node coordinate arrays.

6. **Minor — Masked cell indices silently select their underlying cells.**  
   Location: [jcope_fulldomain.py:401](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:401).

   **Evidence:** On the synthetic day source:

   ```python
   src.read_profile("TT", np.ma.masked_array([3], mask=True), [6])
   ```

   succeeds and returns the finite column at `(3,6)`. `np.asarray` discards the index mask before validation, so an unavailable location becomes an apparently valid selection.

   **Fix:** Reject masked entries in both index arrays before converting them, consistently with node and geographic-coordinate validation.

7. **Minor — Both writers still accept finite values that become missing NetCDF values.**  
   Locations: [jcope_obc_generator.py:946](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:946), [jcope_obc_generator.py:1020](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:1020).

   **Evidence:** Filling otherwise valid inputs with `netCDF4.default_fillvals["f4"]` passes float32 finite validation. Both public writers return successfully, but netCDF4 reads the resulting forcing variables as entirely masked because the variables use that default fill marker.

   This also matters when callers have already discarded a masked array’s mask and exposed its finite fill payload.

   **Fix:** Reject the output variable’s reserved fill marker before publication, or explicitly use a nonfinite fill value that the existing validation already excludes.

8. **Minor — An incomplete damaged-level range is silently narrowed to one level.**  
   Location: [jcopetda/tools/make_provisional_tree.py:176](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:176).

   **Evidence:** `parse_levels("UT:2-")` returns `{"UT": {2}}`. The parser replaces an absent upper endpoint with the lower endpoint even when a range separator was supplied. A truncated range declaration can therefore leave intended damaged levels untreated without an error.

   **Fix:** Distinguish `VAR:N` from `VAR:LO-HI`; require both endpoints whenever `-` is present.

9. **Minor — Source-time validation can discard nonfinite components before checking finiteness.**  
   Location: [run_jcope_obc.py:178](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:178).

   **Evidence:** The following input passes `check_time_axis`:

   ```python
   np.array([
       complex(58849, np.nan),
       complex(58849 + 1/24, np.inf),
   ])
   ```

   Conversion to float64 discards the imaginary components with a warning; the subsequent finite and hourly checks see ordinary valid timestamps. Normal generator-produced MJD arrays are unaffected.

   **Fix:** Reject non-real dtypes before casting, then preserve masks and validate finiteness.

## Verdict
VERDICT: FAIL (0 blocker, 0 major, 9 minor, 0 nit)

### Triage

All 9 findings (minor) accepted. xfvcom 56b5870: #2 (writers keep unmasked float32 without a float64 round trip; measured extra peak 72 MB -> 4.3 MB on a 24 MB input), #3 (mesh node coordinates placed by validated record id), #4 (from_dataset honours start_index), #5 (1-D shapes), #6 (masked indices), #7 (netCDF fill marker rejected), #9 (complex/object times). jcopetda 6698843: #1 (declared levels add to measured damage), #8 (incomplete ranges are errors). No finding of any severity above minor remained after round 3's fixes.
