// robust/cells_wedges.rs — the wedge links of the cell complex: for one
// arrangement edge, which (piece, side) nodes its walls and wedges join.
//
// A child of robust/cells.rs (split out to keep that file under the size
// limit). `build_cells` maps `wedge_links` over chunks of edges in parallel
// and replays the links into its union-find in edge order; the sizes of that
// sweep are here beside the function they batch.

use super::super::exact::approx::orient3d_a;
use super::super::exact::Sign;
use super::super::intersection_graph::{EdgeKey, IntersectionGraph};
use super::{ccw_side, cw_side, node, radial_fan, VertTables};

/// Arrangement edges per work item of the wedge linking: enough that one
/// item's link list amortises its allocation (the per-edge allocations this
/// stage once made dominated it), few enough to balance across workers.
pub(super) const WEDGE_CHUNK: usize = 1024;
/// Chunk count from which the wedge linking runs in parallel.
pub(super) const WEDGE_PAR_THRESHOLD: usize = 2;
/// Chunks mapped per replay window of the wedge linking (bounds the links
/// held at once to this many chunks' worth).
pub(super) const WEDGE_WINDOW: usize = 64;

/// The side-node unions one arrangement edge contributes, appended to `out`
/// in the order they must be applied: every wall's faces share their cell on
/// each side, and each wedge joins one wall's CCW side to the next wall's CW
/// side. `raw` is the edge's incident half-faces (one key).
pub(super) fn wedge_links(
    graph: &IntersectionGraph,
    vt: VertTables,
    raw: &[(EdgeKey, usize, bool, u32)],
    out: &mut Vec<(u32, u32)>,
) {
    if raw.len() < 2 {
        return; // a boundary edge bounds no wedge
    }
    let key = &raw[0].0;
    // Ordinary manifold edge whose two faces are provably in different
    // radial directions: the cyclic order is trivial, so both wedges
    // link with no exact angle work at all. These edges outnumber the
    // rest by a wide margin, and computing rational radial directions
    // for them dominated cell construction. The filter must certify
    // non-coplanarity — a coincident pair has one radial position, not
    // two, and linking it as two would fuse the sheet's own sides.
    if raw.len() == 2 {
        let pt = |v: u32| {
            let p = graph.verts_f64[v as usize];
            [p.x, p.y, p.z]
        };
        let sign = orient3d_a(pt(key.0), pt(key.1), pt(raw[0].3), pt(raw[1].3));
        if matches!(sign, Some(Sign::Neg) | Some(Sign::Pos)) {
            let (p0, fw0) = (raw[0].1, raw[0].2);
            let (p1, fw1) = (raw[1].1, raw[1].2);
            out.push((node(p0, ccw_side(fw0)), node(p1, cw_side(fw1))));
            out.push((node(p1, ccw_side(fw1)), node(p0, cw_side(fw0))));
            return;
        }
    }
    let Some((incs, groups)) = radial_fan(key.0, key.1, raw, vt) else {
        return;
    };
    for gi in 0..groups.len() {
        let (s, e) = groups[gi];
        // All faces of a wall share the cell on each of its two sides.
        for k in s..e {
            out.push((
                node(incs[s].id, incs[s].ccw_side()),
                node(incs[k].id, incs[k].ccw_side()),
            ));
            out.push((
                node(incs[s].id, incs[s].cw_side()),
                node(incs[k].id, incs[k].cw_side()),
            ));
        }
        // The wedge between this wall and the next: CCW side of this
        // wall meets the CW side of the next.
        let (ns, _) = groups[(gi + 1) % groups.len()];
        if groups.len() > 1 {
            out.push((
                node(incs[s].id, incs[s].ccw_side()),
                node(incs[ns].id, incs[ns].cw_side()),
            ));
        }
    }
}
