# Mesh pairs and bounded field transfer

`xfvcom.regrid` transfers fields from a base FVCOM mesh to a mesh produced by
local refinement, using an old-to-new node map. All arrays and maps are
zero-based; `-1` means a removed node or an unmatched element. `Mesh` stores
projected coordinates, counterclockwise triangles and positive-down depths.
`mesh_pair` reads FVCOM grd/dep/cor/obc/spg files through the shared xfvcom
readers, validates maps and topology, classifies frozen/new nodes and elements,
and supplies footprint polygons, edges, r-factors and one-based re-indexing.

Mapped nodes are frozen. Elements are matched by mapped vertex triples, with
orientation preserved; retaining three nodes alone does not retain an element.
Matched data are copied exactly, including dtype and signed zeros. New node
values use barycentric stencils in actual base triangles. New element values
combine barycentric weights with incident triangle area shares at each vertex.
Outside-water queries use donors connected by target mesh edges, with a distance
limit in metres and deterministic lowest-ID ties; they never jump across land.

`build_transfer` builds both CSR operators and horizontal reports;
`build_node_operator` and `build_element_operator` apply their own distance
limits separately. `apply_2d` applies horizontal stencils. `apply_3d` first
remaps each donor in physical z using a common decreasing sigma interface vector
from 0 to -1 and each mesh's depth and elevation, then combines donor values.
It continues endpoint layer values beyond the donor layer centres. Statistics
separate clamping within the donor water column from extension beyond its
surface/bottom. They report donor-weighted target volumes, fractions, distance
quantiles and maxima, and the change that linear extrapolation would produce.
Copy rows are excluded. `ExtensionWarning` reports extension beyond 1 m, drawing
attention to the difference between refined and base depths.

The physical input range is enforced: float64 projected coordinates with
`|x|, |y| <= 1e8` m, edges at least `1e-6` m, triangle areas at least `1e-12` m²,
depths from `1e-3` to `1e5` m, elevations with magnitude at most `1e3` m, and
sigma layer thickness at least `1e-6` of the column. Fields are finite
float32/float64 with magnitude at most `1e100`, spatial axis first. Geometry
whose areas or weights cannot be certified to `1e-12` is refused.

`transfer_reference.independent_evaluate` independently searches triangles,
walks water connections, forms area shares and remaps vertical profiles. It
uses exact rational elimination for geometry and shares only input validation
with `transfer`. `check` compares every output value against this evaluator,
using a per-value round-off allowance; it can also check a supplied result.

The implementation modules use relative imports and can be loaded individually
from source bytes under a private package without executing `xfvcom/__init__.py`.
Application manifests, publication receipts, bundle paths and provenance belong
to the consuming application.
