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

## Round 5 (commit 5fbc3bf reviewed; fixes in eece081)

### Final answer (verbatim)

Reviewed xfvcom `5fbc3bf` and jcopetda `6698843`. No files were modified.

Validation included focused existing tests, diskless NetCDFs, filesystem mocks, and a real one-day build at the 13 production OBC nodes. That build produced finite forcing; all five raw variables matched regional data within packing tolerance. The complete suite and six-year job were not rerun.

**Round-4 status**

| # | Status | Verification |
|---|---|---|
| 1 | RESOLVED | Measured extreme damage and declared levels are combined. The reproduction replaces both outliers on the undeclared damaged level. |
| 2 | RESOLVED | Suitable float32 arrays share memory without conversion copies. Both writers preserved ordinary, read-only and strided caller arrays; raw-profile overrides also remained unchanged. |
| 3 | RESOLVED | Reordered node records populate coordinates by ID; duplicate and out-of-range IDs are rejected. |
| 4 | RESOLVED | Explicit indexing argument and attribute are honored; exported connectivity carries `start_index=0`. Other reconstruction defects appear below. |
| 5 | RESOLVED | Two-dimensional depth and geographic-coordinate arrays are rejected before interpolation. |
| 6 | RESOLVED | Masked entries in either cell-index array are rejected. |
| 7 | RESOLVED | Both writers reject the reserved marker in the reported forcing fields. Depth metadata has a separate omission below. |
| 8 | RESOLVED | `UT:2-` now raises instead of becoming `{2}`. |
| 9 | RESOLVED | Complex and object time arrays are rejected before conversion. |

No caller-buffer mutation regression was reproduced from the float32 change. The following defects remain in the broader scope.

1. **Major — Control-volume geometry depends on the coordinate origin.**  
   Location: [xfvcom/grid/grid_obj.py:690](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:690).

   **Reproduction:** Construct two triangles with:

   ```python
   x = 400000 + 5 * np.array([0., 1., 0., 1.])
   y = 3900000 + 5 * np.array([0., 0., 1., 1.])
   nv = np.array([[0, 1, 2], [1, 3, 2]]).T
   ```

   Triangle areas total **25 m²**, but `calculate_node_area_median_dual()` returns **20.833251953125 m²**. Moving the same mesh to the origin returns 25 m². Default `np.allclose` tolerances consider distinct boundary midpoints equal at UTM coordinate magnitudes, so a required vertex is removed.

   **Fix:** Remove the final midpoint only when topology establishes an interior closure, using equality or a small absolute tolerance with `rtol=0`. Translate polygon coordinates before the shoelace calculation to reduce cancellation. Add translated-mesh coverage. The supplied Tokyo Bay mesh did not exhibit this error in the checked boundary-node areas.

2. **Minor — Three-element connectivity is silently misinterpreted after transposition.**  
   Location: [xfvcom/grid/grid_obj.py:124](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:124).

   **Reproduction:** Export a grid whose triangle rows are `[[0,1,2], [1,3,2], [2,3,4]]`, transpose the dataset so `nv.dims == ("nele", "three")`, then call `from_dataset`. Its `(3,3)` shape prevents transposition. Loaded triangles become `[[0,1,2], [1,3,3], [2,2,4]]`. In the reproduced geometry, areas change from `[1,2.5,1]` to `[1,0,0]`, despite explicit `start_index=0`.

   **Fix:** Orient connectivity using dimension names. Validate the resulting shape; use shape inference only for unnamed inputs whose orientation is unambiguous.

3. **Minor — The explicitly zero-based `nv_zero` fallback still shifts node IDs.**  
   Location: [xfvcom/grid/grid_obj.py:107](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:107).

   **Reproduction:** A five-node dataset containing only `nv_zero=(("nele","three"), [[1,2,3]])` produces triangle `[0,1,2]` through `from_dataset(ds, validate=False)`. Its known zero-based semantics are lost when renamed to `nv`, after which minimum-based inference subtracts one.

   **Fix:** Preserve `start_index=0` when adopting `nv_zero`, unless the caller explicitly overrides it.

4. **Minor — Fractional connectivity is truncated before validation.**  
   Location: [xfvcom/grid/grid_obj.py:123](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:123).

   **Reproduction:** Connectivity `[[0.9],[1.9],[2.9]]` with `start_index=0` is accepted as `[[0],[1],[2]]`, producing finite geometry. Bounds checks cannot detect the discarded fractions.

   **Fix:** Validate the original connectivity’s real numeric type, finiteness, integrality and shape before converting to integers.

5. **Minor — Grid round trips discard projection metadata.**  
   Location: [xfvcom/grid/grid_obj.py:217](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:217), also `from_dataset` at line 144.

   **Reproduction:** For a grid with `zone=54, northern=False`, `FvcomGrid.from_dataset(g.to_xarray())` returns `zone=None, northern=True`. Neither export nor reconstruction preserves these public fields. Previously supplied geographic coordinates may survive, concealing the lost projection information.

   **Fix:** Serialize and restore the UTM zone and hemisphere, preferably through explicit CRS metadata. Include southern-hemisphere round trips in coverage.

6. **Minor — DAT element-record identifiers remain ignored.**  
   Location: [xfvcom/grid/dat_reader.py:76](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/dat_reader.py:76).

   **Reproduction:** A two-element file with records ordered `2 …`, then `1 …`, is accepted and stores element 2 in column zero. Two records both identified as element 1 are also accepted. Element-numbered operations such as `calculate_element_area([1])` consequently address row position instead of the declared element.

   **Fix:** Validate element IDs for range and uniqueness and populate connectivity by ID, matching the corrected node-record handling.

7. **Minor — The regional API can reinterpret undecoded numeric time as nanoseconds.**  
   Location: [xfvcom/io/jcope_obc_generator.py:275](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:275).

   **Reproduction:** Give the regional dataset numeric Unix-second coordinates `1577836800 + 3600*np.arange(24)` without CF time units. Construction succeeds, generates finite fields, and interprets the first time as approximately **1970-01-01 00:00:01.577836544**, with microsecond-scale intervals. `_validate_time` accepts the resulting increasing MJD axis.

   The runner’s year/hour checks protect its normal CLI path; direct generator/writer use remains exposed.

   **Fix:** Require decoded datetime coordinates before constructing `DatetimeIndex`, then validate nonmissing, ordered times. Reject undecoded numeric coordinates explicitly.

8. **Minor — Square static grids can silently transpose bathymetry and ocean masks.**  
   Location: [xfvcom/io/jcope_grid.py:66](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:66).

   **Reproduction:** On an 8×8 grid, transpose `h` and `mask` to dimensions `("lon","lat")`, preserving their named-coordinate meaning. Construction still passes because validation checks shape only. An ocean lookup that selected `(j=3,i=6)` then selects `(j=5,i=6)`. Constant valid bathymetry allows this change through geometry validation.

   This matters for the production grid’s square 1190×1190 shape: shape checks cannot establish dimension order.

   **Fix:** Validate dimension names and normalize these variables with `.transpose("lat", "lon")` before extracting arrays, or explicitly reject unsupported ordering.

9. **Minor — The reserved fill marker is still accepted as boundary depth.**  
   Location: [xfvcom/io/jcope_obc_generator.py:904](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:904).

   **Reproduction:** With otherwise valid inputs, call `write_tsobc_nc` using `obc_h=[netCDF4.default_fillvals["f4"]]`. The writer succeeds, but netCDF4 reads `obc_h` as masked. Its finite-positive check admits the reserved marker.

   **Fix:** Reject the reserved float32 marker in `obc_h` before publication, consistently with forcing-field validation.

10. **Minor — Reading the umask temporarily changes process-wide permissions.**  
    Location: [xfvcom/io/jcope_obc_generator.py:838](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:838).

    **Evidence:** `_current_umask` calls `os.umask(0)` before restoring the previous value. An injected interleaving with an original mask of `077` shows another thread’s ordinary file creation receiving mode **0666 instead of 0600**. Every successful atomic writer reaches this window.

    **Fix:** Avoid changing process-wide umask during writes. Retain restrictive temporary-file permissions or apply an explicit permission policy; a writer-local lock cannot protect unrelated threads.

11. **Minor — Source comparison permits an absent vertical coordinate.**  
    Location: [scripts/compare_jcope_sources.py:69](/home/pj24001722/ku40000343/Github/xfvcom/scripts/compare_jcope_sources.py:69).

    **Reproduction:** `check_alignment(src, grid, reg.drop_vars("level"))` returns all 24 selected positions successfully. Vertical validation runs only when the coordinate exists, although the generator requires it. Thus numerical comparison can proceed without establishing the regional level ordering.

    **Fix:** Require `level` on its own dimension and verify exactly `0..45`, sharing the generator’s regional-schema validation where practical.

## Verdict

VERDICT: FAIL (0 blocker, 1 major, 10 minor, 0 nit)

### Triage

All 11 findings accepted (1 major pre-existing in grid_obj.calculate_node_area_median_dual: area depended on the coordinate origin; the TokyoBay per-node areas agree with the old function to 8.4e-9 relative and their total now equals the total triangle area). Fixed in xfvcom eece081 with 41 new tests (37 failed on the old code): #1 topology-based midpoint removal and translated shoelace; #2/#4 connectivity orientation by dimension names, integral/finite check; #3 nv_zero keeps zero base; #5 utm_zone/hemisphere round trip; #6 dat element ids; #7 decoded region time; #8 JcopeGrid dimension order; #9 fill marker in obc_h/time; #10 umask read from /proc, never set; #11 compare script shares the generator's regional schema validation. jcopetda: nothing in this round.

## Round 6 (commit a1b5d00 reviewed; fixes in 0d0102e, jcopetda 61ad7e8)

### Final answer (verbatim)

Reviewed xfvcom `a1b5d00` and jcopetda `6698843`. No files were modified.

Validation included 30 selected file-free regression tests, in-memory NetCDF round trips, mocks and fault injections, and a real 2020-01-01 build at all 13 production OBC nodes. Regional and full-domain values agreed within packing tolerance; generated forcing was finite. The complete suite and six-year build were not rerun.

**Round-5 status**

| Finding | Status | Verification |
|---|---|---|
| 1 | RESOLVED | Translated meshes retain their areas. Tokyo Bay’s dual-area total agrees with its triangle-area total. |
| 2 | RESOLVED | Named, transposed `(3,3)` connectivity reconstructs correctly. |
| 3 | RESOLVED | Adopted `nv_zero` retains zero-based indexing; explicit overrides work. |
| 4 | RESOLVED | Fractional, nonfinite and non-real connectivity is rejected before conversion. |
| 5 | RESOLVED | Zone and southern-hemisphere metadata survive reconstruction and serialized round trips. |
| 6 | RESOLVED | Reordered element records populate columns by ID; duplicate and out-of-range IDs are rejected. |
| 7 | RESOLVED | Numeric, missing, duplicate and decreasing regional timestamps are rejected. |
| 8 | RESOLVED | Transposing square-grid `h` and `mask` preserves lookup results. |
| 9 | RESOLVED | Depth and time fill markers are rejected before publication. |
| 10 | RESOLVED | Reading the mask no longer calls `os.umask`. A separate fallback regression appears below. |
| 11 | RESOLVED | Comparison requires the regional schema, including the vertical coordinate. |

**Findings**

1. **Major — Grid consumers still corrupt explicitly zero-based connectivity.**  
   Locations: [io/core.py:121](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/core.py:121), [input_loader.py:179](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/input_loader.py:179), [met_nc_generator.py:558](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/met_nc_generator.py:558). Pre-existing.

   **Reproduction:** Serialize `FvcomGrid.to_xarray()` with triangles `[[0,1,2], [1,3,2]]`, then read it through `FvcomDataLoader(utm2geo=False)`. Although `nv.start_index` remains `0`, the loader produces:

   ```text
   nv_zero = [[-1,0,1], [0,2,1]]
   ```

   For node values `[10,20,30,1000]`, subsequent node-to-element averaging returns `[343.333333,20]` instead of `[20,350]`. Negative indices silently sample the last node.

   The input loader and meteorological writer have the related minimum-based inference problem: zero-based connectivity omitting node zero is treated as one-based. A reproduced input-loader case changed triangles `[1,2,3]` and `[2,4,3]` into `[0,1,2]` and `[1,3,2]`.

   **Fix:** Share connectivity normalization across consumers, honoring dimension names and `start_index`. Connectivity obtained directly from `FvcomGrid` is always zero-based and should be converted explicitly.

2. **Major — Geographic fallback silently returns degree-squared areas as square metres.**  
   Location: [grid_obj.py:183](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:183). Pre-existing.

   **Reproduction:** Pass `get_grid()` a dataset containing only geographic coordinates and connectivity:

   ```python
   lon = [139.0, 139.01, 139.0]
   lat = [35.0, 35.0, 35.01]
   nv = [[1], [2], [3]]
   ```

   Its automatic loose fallback copies longitude/latitude into `x/y`. Both triangle and median-dual area calculations then return approximately **0.00005**, documented as m². The geodesic triangle area is approximately **506,378.5 m²**. Construction and calculation succeed without identifying the unit mismatch.

   **Fix:** Project geographic coordinates into metres before constructing a metric grid, or retain coordinate-type metadata and reject metric area calculations on geographic coordinates.

3. **Minor — Three-element meshes remain misoriented in the sigma-z loader.**  
   Location: [gtsz_builder.py:86](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz_builder.py:86). Pre-existing.

   **Reproduction:** `read_dat()` always returns `(3, nele)`, but `load_mesh()` skips transposition when `nele == 3`. Intended triangle rows:

   ```text
   [[0,1,2], [1,3,2], [2,3,4]]
   ```

   become:

   ```text
   [[0,1,2], [1,3,3], [2,2,4]]
   ```

   With `x=[0,1,0,1,0]`, `y=[0,0,1,1,2]`, depths `[100,100,100,100,5]` and `smax=10`, the resulting reference depth at zero-based node 3 is **39.142 m**, versus **19.142 m** with correct connectivity. This reaches the runner’s `--sigma-dat` path.

   **Fix:** Unconditionally transpose connectivity received from `read_dat()`, whose orientation is already known.

4. **Minor — The new umask fallback can widen output permissions.**  
   Locations: [jcope_obc_generator.py:885](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:885), [jcope_obc_generator.py:906](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:906). Round-5 regression.

   **Evidence:** Fault-injecting an unavailable `/proc/self/status` makes `_current_umask()` return `022`. The writer consequently applies mode **0644**, including when the effective process mask is **077**, which requires **0600**. This affects platforms without Linux procfs and environments where procfs is unavailable.

   The unconditional chmod also broadens an existing `0600` destination to `0644` under an ordinary `022` mask.

   **Fix:** Preserve restrictive temporary-file permissions when the mask is unknown. Preserve an existing destination’s permissions when replacing it, unless an explicit permission policy requests otherwise.

5. **Minor — Source-year aliases can still destroy archive data during provisional-tree rebuilding.**  
   Location: [make_provisional_tree.py:247](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:247). Pre-existing.

   **Reproduction:** Use separate roots `/archive` and `/output`, but make `/archive/2020` a symlink to `/output/2020`, containing a valid ordinary day file. The destination-directory guard passes.

   A filesystem-mocked execution of `main()` replaced that day with a symlink pointing to `/archive/2020/<same-day>`. Through the source-year alias, this points back to itself, replacing the last good payload. The command reported `1 links, 0 filled copies` successfully.

   **Fix:** Before cleanup or publication, reject source-year directories resolving inside the output tree or aliasing destination directories.

6. **Minor — Geographic control-volume polygons are displaced by the approximate conversion.**  
   Location: [grid_obj.py:979](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:979). Pre-existing.

   **Reproduction:** On a valid two-triangle, 1 km square mesh at UTM `(400000,3900000)`, zone 54, populate geographic coordinates using the existing projection helper. Comparing `get_node_control_volumes([1])` with the proper transformation of `get_node_control_volume(1)` gives vertex-position errors up to **54.9 m**.

   Each vertex is anchored to its nearest node and shifted using the fixed coefficient `1e-5 degrees/metre`, which is not the grid’s projection.

   **Fix:** Transform polygon vertices with `utm_to_lonlat()` using the grid’s zone and hemisphere. Handle missing projection metadata explicitly.

7. **Minor — Editing connectivity leaves cached topology stale and changes computed areas.**  
   Location: [grid_obj.py:829](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:829). Pre-existing.

   **Reproduction:** Construct a six-triangle fan of radius 10, calculate its central-node area, then remove one triangle through the public `nv` field. The existing object returns **68.560344 m²** for the central node; a fresh object with identical current arrays returns the correct **72.168784 m²**. Cached boundary edges still describe the original mesh.

   **Fix:** Invalidate topology caches whenever connectivity changes, including in-place changes, or make connectivity immutable and provide an update API that invalidates both caches.

8. **Minor — Area selectors silently truncate fractions, discard masks and double-count duplicate nodes.**  
   Location: [grid_obj.py:899](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/grid_obj.py:899). Pre-existing; similar coercion occurs in the other selection methods.

   **Reproduction:** On a unit right triangle:

   - `[1.9]` silently selects node 1.
   - A masked selector `[1, --]` with underlying values `[1,2]` includes node 2.
   - `[1,1]` returns twice node 1’s median-dual area, unlike `calculate_node_area()`, which treats node selection as a set.

   **Fix:** Validate selectors before integer conversion: reject masked and fractional entries, validate the indexing base, and reject or deduplicate repeated node IDs.

9. **Minor — Missing static-mask values become ocean cells.**  
   Location: [jcope_grid.py:78](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_grid.py:78). Pre-existing.

   **Reproduction:** Supply a decoded mask containing:

   ```text
   [[NaN, 1],
    [0,   0]]
   ```

   Construction succeeds and converts it to:

   ```text
   [[True,  True],
    [False, False]]
   ```

   A lookup at the missing cell selects that cell as ocean. NaN can arise normally when xarray decodes a NetCDF fill value; dimension normalization does not protect against this conversion.

   **Fix:** Validate the decoded mask as finite, unmasked binary values before converting it to Boolean. Reject missing or unsupported flags explicitly.

10. **Minor — Shared element-to-node interpolation fails on integer fields.**  
    Location: [interpolation.py:114](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/interpolation.py:114). Pre-existing.

    **Reproduction:** Using the valid two-triangle connectivity from finding 1:

    ```python
    elems2nodes(np.array([1, 2]), grid.nv, n_nodes=4)
    ```

    raises `UFuncTypeError` during in-place division because the accumulator retains integer dtype. The expected result is `[1,1.5,1.5,2]`. The same problem affects the supported two- and three-dimensional integer inputs.

    **Fix:** Promote the accumulator to a floating result dtype before accumulation and division; cover all three supported input ranks.

## Verdict

VERDICT: FAIL (0 blocker, 2 major, 8 minor, 0 nit)

### Triage

All 10 findings accepted. Only #4 (umask fallback) and #5 (source-year alias in the jcopetda tool) were defects of this work; #1 #2 #3 #6 #7 #8 #9 #10 are pre-existing defects of general xfvcom modules (the round-5/6 prompts asked for 'everything it touches', which pulled the whole grid library in). Fixed in 0d0102e (31 tests, 27 failed on the old code): shared connectivity normalisation (#1), geographic grids are marked and metric areas raise (#2: marked, not projected), gtsz load_mesh transposes always (#3), writer permission policy (#4), control-volume vertices via the grid's projection (#6), topology caches keyed on an nv fingerprint (#7), strict selectors (#8), JcopeGrid mask validation (#9), elems2nodes float accumulator (#10). jcopetda 61ad7e8: #5. Behaviour changes for existing callers are listed in the commit message of 0d0102e. From round 7 on the review scope is stated explicitly (the modules run_jcope_obc.py reaches), because the free audit had left the subject of this work.

## Round 7 (commit 98e74bf reviewed, scope stated explicitly; fixes in 294c361, jcopetda 41c46e4)

### Final answer (verbatim)

Reviewed xfvcom `98e74bf` and jcopetda `61ad7e8`. No files were modified.

Validation included 17 existing file-free regression cases, in-memory NetCDF and filesystem mocks, and real 2020-01-01 data at all 13 production OBC nodes. All five raw variables matched the regional archive within packing tolerance. Uniform-sigma, SADAPT and depth-control builds produced finite forcing. The complete suite and six-year build were not rerun.

**Round-6 status**

| Finding | Status | Verification |
|---|---|---|
| 1. Connectivity consumers | RESOLVED | Core loader, input loader and meteorological writer preserve explicitly zero-based connectivity, including meshes omitting node zero. |
| 2. Geographic areas | RESOLVED | All three metric-area methods reject geographic grids; the geographic flag survives reconstruction. |
| 3. Three-element sigma-z meshes | RESOLVED | `load_mesh()` returns the intended triangle rows for a three-element mesh. |
| 4. Writer permissions | RESOLVED | Replacements retain existing permission bits; unknown umask retains `0600`; known masks produce the expected new-file modes. |
| 5. Source-year aliases | RESOLVED | The reported source-year alias is rejected before day publication. A separate day-file alias remains: finding 5 below. |
| 6. Geographic control volumes | RESOLVED | Projection-based vertices agree with independently transformed vertices in both hemispheres. |
| 7. Connectivity cache invalidation | RESOLVED | Replacement and in-place connectivity edits produce the same topology and areas as fresh objects. |
| 8. Selectors | RESOLVED | Reported fractional, masked, Boolean and invalid-base selectors are rejected; duplicate-node area sums count nodes once. |
| 9. Static mask | RESOLVED | Missing and nonbinary decoded masks are rejected; valid binary masks load. |
| 10. Integer interpolation | RESOLVED | Integer inputs work across supported ranks; Boolean interpolation works and float32 output retains its dtype. |

**Findings**

1. **Major — `--siglev-npz` ignores the node identities and bathymetry accompanying the coordinate.**  
   Location: [scripts/run_jcope_obc.py:648](/home/pj24001722/ku40000343/Github/xfvcom/scripts/run_jcope_obc.py:648). Pre-existing.

   **Evidence:** The production `lsc2_obc_siglev_kb51.npz` contains `siglev`, `obc_node` and `h`, but the runner reads only `siglev` and checks the node count. Reversing the OBC order therefore passes validation while assigning coordinates to different nodes. The equivalent column-swap check using production-day profiles changed temperature by up to **7.7488°C**, with finite output throughout.

   The same NPZ is accepted with `TokyoBay_dep.dat`, although its recorded depths differ by up to **347.18 m**. Its intended matching depth file agrees within approximately `6e-6 m`.

   **Fix:** Validate supplied node IDs and reorder coordinates explicitly by ID, or reject order mismatches. Validate recorded depths against the selected OBC depths. Require sufficient identity metadata to establish that externally built coordinates belong to the requested boundary.

2. **Major — Sigma-file numeric parsing can construct a different coordinate from the one FVCOM actually uses.**  
   Location: [xfvcom/io/sigma_dat.py:119](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:119). Pre-existing.

   **Reproduction:** Read a pure SIGMAZ file containing:

   ```text
   NUMBER OF SIGMA LEVELS = 31
   SIGMA COORDINATE TYPE = SIGMAZ
   GTSZ BASE = 2
   GTSZ P1 = 4
   ```

   Python uses `P1=4`. FVCOM’s scanner classifies decimal-less `4` as an integer, rejects the floating-scalar request, and [falls back to `P1=2`]( /home/pj24001722/ku40000343/Github/FVCOM/src/mod_input.F:5728). The resulting interfaces differ by up to **37.29 m** in a 300 m column. Using the production-day profiles, the resulting temperature fields differed by **2.8824°C**, while both passed forcing validation.

   This is reachable through ordinary library-generated files: `gtsz_to_lines()` formats nondefault integral-valued floating parameters with `:g`, producing precisely this syntax.

   **Fix:** Make the reader reproduce FVCOM’s effective parameter semantics, or reject forms that FVCOM interprets differently. Emit decimal literals for floating parameters.

3. **Minor — LOWFRAC coordinates omit FVCOM’s normalization step.**  
   Location: [xfvcom/grid/gtsz.py:385](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz.py:385). Pre-existing.

   **Reproduction:** Use `kb=5`, `base=1`, `k1=2`, `sadapt=True`, `lowsig=True`, and `lowfrac=(0.33331, 0.33331, 0.33331)`. Their sum, `0.99993`, passes both implementations’ tolerance checks. FVCOM [normalizes the fractions]( /home/pj24001722/ku40000343/Github/FVCOM/src/mod_input.F:5774); Python accumulates the original fractions.

   With `H=300 m` and `href=20 m`, the resulting interfaces differ by approximately **13.1 mm**. Thus the runner’s supposedly matching per-node coordinate differs systematically from the model’s coordinate.

   **Fix:** Normalize accepted LOWFRAC values before accumulating them, consistently with FVCOM.

4. **Minor — GTSZ validation accepts model-invalid parameters that produce plausible finite coordinates.**  
   Location: [xfvcom/grid/gtsz.py:129](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz.py:129). Pre-existing.

   **Reproduction:** Both `GtszSpec(kb=31, p1=0.0)` and `GtszSpec(kb=31, smooth=-1.0)` pass `validate()`. Their constructed columns also pass the generator’s sigma validation.

   With `P1=0`, most interfaces cluster near sigma `-0.5`; negative smoothing silently selects the unsmoothed branch. FVCOM explicitly rejects these settings at startup. Corresponding decimal-valued `sigma.dat` entries reach these paths through the runner.

   **Fix:** Mirror FVCOM’s relevant startup checks, including finite positive A2 `P1`, finite nonnegative `SMOOTH`, and the SADAPT surface-band constraints.

5. **Minor — Individual source-day aliases can still destroy the last good payload during provisional rebuilding.**  
   Location: [jcopetda/tools/make_provisional_tree.py:275](/home/pj24001722/ku40000343/Github/jcopetda/tools/make_provisional_tree.py:275). Pre-existing.

   **Reproduction:** Keep both year directories ordinary directories, but arrange:

   ```text
   /archive/2020/jcopetda_20200101.nc -> /output/2020/jcopetda_20200101.nc
   /output/2020/jcopetda_20200101.nc  = valid ordinary day file
   ```

   A filesystem-mocked execution of the actual `main()` passed both directory guards, replaced the output payload with a link back to the source alias, and reported `1 links, 0 filled copies`. Both paths then formed a symlink loop; the payload was gone.

   **Fix:** Preflight individual source-day paths before reconciliation. Reject files resolving inside the output tree or aliasing destination entries.

6. **Minor — Exhausting the grassfire sweep budget returns an unconverged reference surface as success.**  
   Location: [xfvcom/grid/gtsz_builder.py:167](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz_builder.py:167). Pre-existing.

   **Reproduction:** On the production Tokyo Bay mesh and `TokyoBay_dep_m7001tp_rfac0p2_cap300.dat`, use the production SADAPT specification’s slope and inject `max_sweep=1`. The function returns normally, although its reference depths differ from the converged result by up to **225.18 m**.

   The function never distinguishes convergence from budget exhaustion. The runner uses the default budget, so this requires an unusually demanding mesh or an injected limit; the checked Tokyo Bay default converges.

   **Fix:** Track convergence and raise when the budget is exhausted without satisfying the tolerance. Match FVCOM’s explicit failure behavior.

7. **Minor — The shared connectivity fix silently accepts contradictory indexing-base attributes.**  
   Location: [xfvcom/grid/connectivity.py:104](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/connectivity.py:104). **Round-6 regression.**

   **Reproduction:** Set `nv.attrs["start_index"] = np.array([0, 1])`. The new helper takes only the first entry and silently chooses zero-based indexing.

   For `x=[0,1,0,5]`, `y=[0,0,1,1]`, and connectivity `[1,2,3]`, current `FvcomGrid.from_dataset()` accepts this contradictory metadata and returns triangle area **2.5**. Executing the pre-round-6 implementation against the same dataset rejects it. An empty attribute array now causes an incidental `IndexError`.

   **Fix:** Require exactly one attribute value before validating its base; reject empty and multi-valued attributes explicitly.

## Verdict

VERDICT: FAIL (0 blocker, 2 major, 5 minor, 0 nit)

### Triage

All 7 findings accepted (2 major, 5 minor; all on the sigma-z / siglev-npz / SIGMAZ-file path or the jcopetda tool, pre-existing except #7). xfvcom 294c361 (32 tests, 26 failed on the old code): #1 load_siglev_npz reorders by obc_node and checks the npz depths against the OBC depths (largest difference 347.18 m for the wrong depth file, 6e-6 m for the intended one; --siglev-npz-ignore-depth), #2 the sigma.dat reader types values like FVCOM's SCAN_FILE/GET_VALUE and REJECTS what FVCOM would read differently (e.g. 'GTSZ P1 = 4'), the writer always writes a decimal point (before, ZLEV was written '-2 -4', which FVCOM treats as missing); mirroring was not chosen because several cases are ambiguous; all 51 existing SIGMAZ files under ~/Github still parse to the same spec, #3 LOWFRAC normalised, #4 validate() mirrors FVCOM's startup checks (P1 > 0, SMOOTH >= 0, ...), #6 grassfire raises when it does not converge (the production case converges in 11 sweeps), #7 start_index must hold exactly one value. jcopetda 41c46e4: #5 individual source day files are preflighted.

## Round 8 (commit b154467 reviewed; fixes in 9748876)

### Final answer (verbatim)

No files were modified. Twenty file-free round-7 tests passed. All five variables matched the regional archive within packing tolerance for 2020-01-01 at all 13 production OBC nodes. Uniform, SADAPT, NPZ and depth-control builds produced finite forcing for 2020-12-15. The full suite was not rerun.

**Round-7 status**

| Finding | Status | Verification |
|---|---|---|
| 1. NPZ identity/depth | RESOLVED | Production rows reorder correctly; the wrong depth file is rejected. Missing metadata produces explicit warnings. |
| 2. FVCOM numeric parsing | PARTIAL | Reported decimal-less values are rejected and floating literals are emitted correctly. Findings 2–5 below identify remaining scanner mismatches. |
| 3. LOWFRAC normalization | RESOLVED | The reported fractions produce the same coordinate as explicitly normalized fractions. |
| 4. GTSZ startup validation | PARTIAL | Reported shape-parameter cases are rejected. Additional model-invalid configurations remain accepted; see finding 6. |
| 5. Source-day aliases | RESOLVED | The reported alias is rejected by the actual tool’s preflight in a filesystem mock. |
| 6. Grassfire exhaustion | RESOLVED | Injecting `max_sweep=1` raises; the production mesh converges normally. |
| 7. `start_index` cardinality | RESOLVED | Empty and multivalued attributes are rejected; supported single-valued forms work. |

**Findings**

1. **Minor — Element aperture fractions use an obsolete FVCOM formula.**  
   Location: [gtsz.py:521](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz.py:521). Pre-existing.

   **Evidence:** Build `tb_sz_m7prod_k30_s02_sigma.dat` on `TokyoBay_grd.dat` with `TokyoBay_dep_m7001tp_rfac0p2_cap300.dat`. At one-based element **5309**, Python returns `frac1=0.9160873153`; the current FVCOM depth-weighted formula gives **0.8947629271**. The corresponding apertures are **134.504935 m** and **131.373972 m**.

   Current FVCOM computes `sum(H_i * f_i) / sum(H_i)`, where `f_i=-Z_i[KBP1]`; Python averages the fractions without depth weighting. This executes during boundary generation, although the runner does not use `frac1` to interpolate T/S.

   **Fix:** Pass node depths into `compute_kbp1_frac1()` and reproduce the depth-weighted formula in `FVCOM/src/mod_setup.F`.

2. **Minor — Comma normalization silently turns true logical values into false.**  
   Locations: [sigma_dat.py:183](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:183), [sigma_dat.py:118](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:118). **Round-7 regression.**

   **Reproduction:** Read an otherwise valid SIGMAZ file containing:

   ```text
   GTSZ SADAPT = T,
   GTSZ LOWSIG = T,
   ```

   Both flags become `False`. `_fvcom_type()` replaces commas before recognizing a logical, but `_get_bool()` compares the original `T,` with `T`. FVCOM replaces commas before interpreting the value and reads both flags as true. Python consequently accepts a different coordinate.

   The integer reader has the related inconsistency: `NUMBER OF SIGMA LEVELS = 31,` passes type classification but then fails integer conversion.

   **Fix:** Normalize commas once and use the normalized value consistently for classification and conversion.

3. **Minor — Trimming tabs still permits scientifically different numeric interpretations.**  
   Location: [sigma_dat.py:239](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:239). Pre-existing; round-7 parsing fix remains incomplete.

   **Reproduction:** In a pure SIGMAZ file, set the value to a literal tab followed by `4.0`:

   ```python
   "GTSZ P1 = \t4.0\n"
   ```

   Python strips the tab and uses `P1=4`. FVCOM’s `ADJUSTL` removes spaces, while `GET_VALUE` treats the retained tab as a string character; the floating request fails and uses `P1=2`. The reproduced interface difference reaches **37.2859 m** in a 300 m column.

   **Fix:** Apply FVCOM’s space-only trimming semantics, or explicitly reject tabs and other whitespace that FVCOM interprets differently.

4. **Minor — Coordinate-type capitalization accepts files FVCOM rejects.**  
   Location: [sigma_dat.py:261](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:261). Pre-existing.

   **Reproduction:** Change `SIGMA COORDINATE TYPE = SIGMAZ` to `SIGMA COORDINATE TYPE = sigmaz`. Python uppercases it and successfully builds the coordinate. FVCOM selects the type case-sensitively and terminates through its default case.

   **Fix:** Reject noncanonical type values instead of silently uppercasing them.

5. **Minor — Scanner limits are ignored, including by the expanded floating-value writer.**  
   Locations: [sigma_dat.py:229](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:229), [sigma_dat.py:371](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:371). Reader defect is pre-existing; longer writer output is a **round-7 regression**.

   **Evidence:**
   - Put mandatory keys near the start, then 201 comment lines, then `GTSZ P1 = 4.0`. Python uses 4; FVCOM exhausts its header search and defaults to 2.
   - Serialize `GtszSpec(kb=91, k1=20, k2=91, nz=70, zlev=tuple(-float(i) for i in range(1,71)))`. The new ZLEV line is **422 characters**, exceeding FVCOM’s 400-character physical input buffer. Its former representation was 282 characters.
   - Python also permits numeric lists beyond FVCOM’s 150-value token buffer.

   **Fix:** Validate scanner limits before accepting or emitting a file. Support bounded continuations consistently in both reader and writer where possible; otherwise reject unsupported files explicitly.

6. **Minor — Additional FVCOM startup constraints remain unchecked.**  
   Locations: [gtsz_builder.py:281](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz_builder.py:281), [sigma_dat.py:296](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:296). Pre-existing.

   **Reproductions:**
   - `GtszSpec(kb=5, mask=True)` validates and builds despite having no z-band; FVCOM explicitly rejects this combination.
   - On a constant 10 m triangle, `GtszSpec(kb=5, base=1, k1=2, k2=5, nz=2, zlev=(-1.,-2.), mask=True, sadapt=True)` builds successfully with `kbp=[1,1,1]`. FVCOM rejects masked columns having only one active layer.
   - A SIGMAZ file containing `BPG REF NZ = -1` is accepted and converted to an empty reference list. FVCOM rejects counts outside `0..500`.

   **Fix:** Check the mask/z-band relationship, reject generated masked columns with fewer than two active layers, and validate the BPG count before discarding its representation.

7. **Minor — Minimum layer counts disagree across supported entrypoints and with FVCOM.**  
   Locations: [gtsz.py:133](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz.py:133), [jcope_obc_generator.py:668](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_obc_generator.py:668). Pre-existing.

   **Evidence:** `GtszSpec(kb=3).validate()` rejects a valid two-layer coordinate. FVCOM’s `CHECK_LAYER_COUNT` accepts three levels, and its base coordinate is `[0,-0.5,-1]`.

   Conversely, explicit interfaces `[0,-1]` and centre `[-0.5]` pass the shared sigma validator, so the NPZ path and writer accept a one-layer T/S coordinate that ordinary 3-D FVCOM rejects. The uniform generator already requires two layers.

   **Fix:** Use a consistent minimum of three interfaces/two layers for these 3-D paths.

8. **Minor — Relative archive roots change meaning after construction.**  
   Location: [jcope_fulldomain.py:105](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/jcope_fulldomain.py:105). Pre-existing.

   **Reproduction:** From `/home/pj24001722/share/Data`, construct:

   ```python
   src = FullDomainSource(
       "jcopetda_provisional", "2020-01-01", "2020-01-01"
   )
   ```

   Change directory to the repository, then call `src.read_2d("EGT", [0], [0])`. Construction has successfully inspected the archive, but reading raises `FileNotFoundError` because paths are resolved against the new working directory. Another matching relative tree could instead supply uninspected payloads.

   **Fix:** Resolve the expanded root to an absolute path during construction.

## Verdict
VERDICT: FAIL (0 blocker, 0 major, 8 minor, 0 nit)

### Triage

All 8 findings accepted (all minor; #2 was a round-7 regression, #8 a relative-root bug of FullDomainSource, the rest FVCOM-compatibility details of the sigma.dat reader/writer and GTSZ checks). xfvcom 9748876 (22 tests): #1 depth-weighted FRAC1 as in mod_setup.F (compute_kbp1_frac1 now takes H), #2 commas normalised once, #3 space-only trimming (tabs rejected), #4 case-sensitive coordinate type, #5 scanner limits (201 header lines, 400 characters, 150 values) enforced by reader and writer (the writer shortens list literals so the 70-level ZLEV fits), #6 MASK needs a z-band, masked columns need >= 2 layers, BPG REF NZ in 0..500, #7 one minimum (3 levels / 2 layers), #8 absolute root. All 3756 *sigma*.dat files under ~/Github parse to the same spec as before.

## Round 9 (commit 9748876 reviewed; fixes in b89128d)

### Final answer (verbatim)

Reviewed xfvcom `9748876` and jcopetda `41c46e4`. No files were modified.

The round-8 cases passed using in-memory file adapters and real-archive reads. Production checks covered all five variables against the regional archive and finite uniform, SADAPT, NPZ, and depth-control builds. Both NetCDF writers were checked in memory. The full suite and multi-year build were not rerun.

| Round-8 finding | Status | Evidence |
|---|---|---|
| 1. Depth-weighted FRAC1 | RESOLVED | Production element 5309 gives `0.8947629270902231`, aperture `131.373971799 m`. |
| 2. Comma normalization | RESOLVED | Reported trailing-comma logical and integer cases work. A separate normalization regression remains below. |
| 3. Tabs/CR | RESOLVED | Non-space whitespace in data lines is rejected; comment cases remain accepted. |
| 4. Coordinate-type case | RESOLVED | Noncanonical capitalization is rejected. |
| 5. Scanner limits | RESOLVED | Reported line, header-position and token limits are enforced; compacted lists round-trip. |
| 6. Startup constraints | RESOLVED | All three reported MASK/BPG cases are rejected. Another missing startup check is finding 1 below. |
| 7. Minimum layer count | RESOLVED | Three levels work; one-layer coordinates are rejected. |
| 8. Relative archive root | RESOLVED | A real source constructed with a relative root remains readable after changing directory. |

1. **Minor — LOWSIG builds accept invalid interior columns that FVCOM rejects.**  
   Location: [xfvcom/grid/gtsz_builder.py:280](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/gtsz_builder.py:280). Pre-existing.

   **Reproduction:** Use the production `TokyoBay_grd.dat`, `TokyoBay_dep_m7001tp2_raw.dat`, and `tb_ss30_m7tp_kb31_sigma.dat`. The depth file contains ten nonpositive depths. `build_gtsz()` succeeds, although 111 sigma-layer thicknesses are nonpositive. The OBC columns pass validation, and an actual production-day build returns finite temperature and salinity.

   Thus validation restricted to the boundary hides invalid columns elsewhere in the mesh. FVCOM’s `SIGMA_GTSZ` checks the minimum physical LOWSIG layer thickness over the domain and terminates for this configuration (`FVCOM/src/mod_setup.F:1594`).

   **Fix:** Validate mesh depths before construction and reproduce FVCOM’s domain-wide positive-thickness check for LOWSIG.

2. **Minor — Reordered grid records can pass the runner’s consistency check while describing different node locations to FVCOM.**  
   Location: [xfvcom/grid/dat_reader.py:114](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/grid/dat_reader.py:114). Pre-existing in this review round.

   **Reproduction:** In memory, swap the complete node records for production OBC nodes **3142** and **3016**, retaining their IDs. `FvcomGrid.from_dat()` restores coordinates by ID, and `parse_dep_dat()` accepts the unchanged production depth file.

   FVCOM’s `READ_COLDSTART_COORDS`, however, reads `J,X1,Y1` and assigns coordinates by **row position**, ignoring `J` (`FVCOM/src/mod_input.F:4967`). The resulting location discrepancy at node 3142 is **16,278.87 m**. The generator therefore samples a different physical location from the one FVCOM assigns that node.

   **Fix:** Reject nonsequential node-record IDs on the FVCOM runner path. An explicit permissive mode can retain ID-based reconstruction for other uses.

3. **Minor — Comma normalization now removes leading key characters that FVCOM preserves.**  
   Location: [xfvcom/io/sigma_dat.py:286](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:286). **Round-8 regression.**

   **Reproduction:** Add this to an otherwise valid pure SIGMAZ file:

   ```text
   ,GTSZ P1 = 4.0
   ```

   The current reader uses `P1=4`; the pre-round-8 reader uses the default `P1=2`.

   FVCOM applies `ADJUSTL` **before** replacing commas. Consequently, the comma becomes a leading space in the key, which does not match `GTSZ P1`; FVCOM uses its default. Python replaces commas and then applies `lstrip`, making the key match. The two coordinates differ by up to **37.2859 m** in a 300 m column.

   **Fix:** Preserve FVCOM’s operation order, or reject leading commas explicitly.

4. **Minor — Continued comments can silently hide parameters from FVCOM.**  
   Locations: [xfvcom/io/sigma_dat.py:266](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:266), [xfvcom/io/sigma_dat.py:485](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:485). Pre-existing.

   **Reproduction:** Place a comment ending in four literal backslashes immediately before:

   ```text
   GTSZ P1 = 4.0
   ```

   Python skips the comment and uses `P1=4`. FVCOM processes continuations before comments, joining the parameter onto the comment line and discarding it. It consequently uses `P1=2`.

   The writer also accepts such a `header_comment`; its final comment can swallow the mandatory level-count line.

   **Fix:** Reject supported continuation suffixes before skipping comments, including in writer-supplied comments, unless continuation semantics are implemented explicitly.

5. **Minor — Fixed-decimal serialization changes valid coordinates and can emit invalid fractions.**  
   Locations: [xfvcom/io/sigma_dat.py:508](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:508), [xfvcom/io/sigma_dat.py:519](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:519). Pre-existing.

   **Evidence:** `smax=0.0000125` is emitted as `0.000013`. On the production capped mesh, serializing and reading a valid LOWSIG specification changes interfaces by up to **0.01581768 m**. `smax=4e-7` becomes zero, selecting the flat-reference branch.

   A valid specification with `kb=31`, `k1=3`, `sadapt=True`, `lowsig=True`, and:

   ```python
   lowfrac = (1e-7,) + ((1 - 1e-7) / 27,) * 27
   ```

   emits its first fraction as `0.000000`. FVCOM rejects that nonpositive fraction.

   **Fix:** Use decimal-bearing significant-digit formatting for SMAX, DREF and LOWFRAC, preserving positive values and coordinate parameters through serialization.

6. **Minor — The writer emits single-value vectors that its reader and FVCOM reject.**  
   Location: [xfvcom/io/sigma_dat.py:463](/home/pj24001722/ku40000343/Github/xfvcom/xfvcom/io/sigma_dat.py:463). Pre-existing.

   **Reproduction:** Both specifications validate and serialize successfully:

   ```python
   GtszSpec(kb=4, k1=1, k2=3, nz=1, zlev=(-2.0,))
   GtszSpec(kb=3, bpg_ref_zlev=(0.0,))
   ```

   They emit one-value `GTSZ ZLEV` and `BPG REF ZLEV` lists respectively. `read_sigma_dat()` rejects both. FVCOM’s `SCAN_FILE` likewise returns `-4` for a one-value `FVEC`, causing startup failure.

   **Fix:** Reject unsupported single-value mandatory vectors before writing. Keep this restriction in serialization rather than prohibiting otherwise useful in-memory coordinates.

## Verdict

VERDICT: FAIL (0 blocker, 0 major, 6 minor, 0 nit)

### Triage

All 6 findings accepted (minor; #3 was a round-8 regression; all in the sigma.dat reader/writer, the GTSZ builder or the grid-file reader, none on the uniform-sigma production path). xfvcom b89128d (29 tests): #1 build_gtsz refuses non-finite depths and, with LOWSIG, repeats FVCOM's domain-wide positive-thickness check (the reviewer's m7001tp2_raw case is rejected; all 42 distinct TB-FVCOM SIGMAZ specs still build on the production cap300 depth file), #2 read_dat/from_dat require node AND element ids 1..n in file order by default (FVCOM places them by row position; allow_unordered=True keeps ID placement; the runner is strict), #3 exact FVCOM order (ADJUSTL, then commas), #4 backslash-continuation lines rejected by reader and writer, #5 floats written in shortest exact form with a decimal point, #6 single-value mandatory vectors refused by the writer. All 2,993 *sigma*.dat files found parse identically before/after.
