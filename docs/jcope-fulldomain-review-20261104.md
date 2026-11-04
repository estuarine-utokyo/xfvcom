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
