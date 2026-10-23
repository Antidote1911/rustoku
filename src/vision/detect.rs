//! Locating sudoku grids in a picture and cutting them into cells.

use image::{imageops, imageops::FilterType, GrayImage, Luma};
use imageproc::{
    contrast::adaptive_threshold,
    filter::gaussian_blur_f32,
    geometric_transformations::{warp_into, Border, Interpolation, Projection},
    geometry::{approximate_polygon_dp, arc_length, convex_hull},
    point::Point,
    region_labelling::{connected_components, Connectivity},
};

/// Side of a cell in the rectified grid, in pixels.
pub const CELL: u32 = 64;
pub const SIDE: u32 = 9 * CELL;

/// Corners of a grid: top-left, top-right, bottom-right, bottom-left.
pub type Quad = [(f32, f32); 4];

/// Pictures are analysed at this resolution at most to find grids.
const WORK_SIZE: u32 = 1000;
/// Minimal `line_score` to accept a candidate.
const MIN_LINE_SCORE: f32 = 0.3;

/// Local thresholding, 255 = ink.
pub fn binarize(gray: &GrayImage, block_radius: u32, delta: i32) -> GrayImage {
    let mut bin = adaptive_threshold(gray, block_radius, delta);
    bin.pixels_mut().for_each(|p| p.0[0] = 255 - p.0[0]);
    bin
}

/// Bounding box, size and extreme points of a connected component.
#[derive(Clone)]
pub struct Blob {
    pub x0: u32,
    pub y0: u32,
    pub x1: u32,
    pub y1: u32,
    pub area: u32,
    /// Pixels minimizing x+y, maximizing x-y, maximizing x+y, minimizing x-y.
    extremes: [(u32, u32); 4],
}

impl Blob {
    fn new(x: u32, y: u32) -> Self {
        Blob {
            x0: x,
            y0: y,
            x1: x,
            y1: y,
            area: 0,
            extremes: [(x, y); 4],
        }
    }

    fn add(&mut self, x: u32, y: u32) {
        self.x0 = self.x0.min(x);
        self.y0 = self.y0.min(y);
        self.x1 = self.x1.max(x);
        self.y1 = self.y1.max(y);
        self.area += 1;
        let (s, d) = (x as i64 + y as i64, x as i64 - y as i64);
        let e = &mut self.extremes;
        if s < e[0].0 as i64 + e[0].1 as i64 {
            e[0] = (x, y);
        }
        if d > e[1].0 as i64 - e[1].1 as i64 {
            e[1] = (x, y);
        }
        if s > e[2].0 as i64 + e[2].1 as i64 {
            e[2] = (x, y);
        }
        if d < e[3].0 as i64 - e[3].1 as i64 {
            e[3] = (x, y);
        }
    }

    pub fn width(&self) -> u32 {
        self.x1 - self.x0 + 1
    }

    pub fn height(&self) -> u32 {
        self.y1 - self.y0 + 1
    }
}

/// Connected components of the ink (255) pixels, indexed by label - 1, with the label image.
pub fn blobs(bin: &GrayImage) -> (Vec<Blob>, Labels) {
    let labels = connected_components(bin, Connectivity::Eight, Luma([0u8]));
    let mut blobs: Vec<Option<Blob>> = Vec::new();
    for (x, y, Luma([l])) in labels.enumerate_pixels() {
        if *l == 0 {
            continue;
        }
        let i = *l as usize - 1;
        if blobs.len() <= i {
            blobs.resize(i + 1, None);
        }
        blobs[i].get_or_insert_with(|| Blob::new(x, y)).add(x, y);
    }
    (
        blobs
            .into_iter()
            .map(|b| b.expect("labels are contiguous"))
            .collect(),
        labels,
    )
}

/// Maps the quadrilateral `quad` of `gray` onto a `SIDE` x `SIDE` square.
pub fn rectify(gray: &GrayImage, quad: &Quad) -> Option<GrayImage> {
    let s = SIDE as f32;
    let projection =
        Projection::from_control_points(*quad, [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)])?;
    let mut out = GrayImage::new(SIDE, SIDE);
    warp_into(
        gray,
        projection,
        Interpolation::Bilinear,
        Border::Constant(Luma([255])),
        &mut out,
    );
    Some(out)
}

/// Ink pixel belonging to a thin horizontal (or vertical) stroke, such as a
/// grid line, as opposed to a filled area like a shaded cell.
fn thin_ink(bin: &GrayImage, x: u32, y: u32, horizontal: bool) -> bool {
    let t = CELL / 6;
    let ink = |x: u32, y: u32| x < SIDE && y < SIDE && bin.get_pixel(x, y).0[0] > 0;
    ink(x, y)
        && if horizontal {
            !(y >= t && ink(x, y - t) && ink(x, y + t))
        } else {
            !(x >= t && ink(x - t, y) && ink(x + t, y))
        }
}

/// Contrast between the ink coverage of the 16 inner lines of a rectified grid
/// and the coverage of the middle of the cells: close to 1 for a real sudoku
/// grid, low for any other square or for a misplaced grid.
pub fn line_score(bin: &GrayImage) -> f32 {
    let tolerance = CELL / 8;
    let coverage = |a: u32, horizontal: bool| -> f32 {
        let ink = (0..SIDE)
            .filter(|&b| {
                let (x, y) = if horizontal { (b, a) } else { (a, b) };
                thin_ink(bin, x, y, horizontal)
            })
            .count();
        ink as f32 / SIDE as f32
    };
    let mut lines = 0.0;
    let mut middles = 0.0;
    for horizontal in [true, false] {
        for k in 1..9 {
            let center = k * CELL;
            lines += (center - tolerance..=center + tolerance)
                .map(|a| coverage(a, horizontal))
                .fold(0.0, f32::max);
        }
        for k in 0..9 {
            middles += coverage(k * CELL + CELL / 2, horizontal);
        }
    }
    lines / 16.0 - middles / 18.0
}

/// Offset, spacing and score of the 10 equally spaced lines best supported by an
/// ink profile. A line one step outside the lattice is penalized, otherwise a
/// lattice shifted by one cell onto a neighbouring frame would score as well.
fn fit_lattice(profile: &[f32]) -> (f32, f32, f32) {
    let n = profile.len() as f32;
    let at = |p: f32| -> f32 {
        let i = p.round() as isize;
        (i - 1..=i + 1)
            .filter(|&i| i >= 0 && (i as usize) < profile.len())
            .map(|i| profile[i as usize])
            .fold(0.0, f32::max)
    };
    let mut best = (0.0, n / 9.0, f32::MIN);
    // The grid fills most of the candidate (this also excludes half-cell lattices),
    // but may exceed it a little when its border was not detected.
    let mut spacing = n / 9.0 * 0.6;
    while spacing <= n / 9.0 * 1.15 {
        // At most half a cell outside: a lattice shifted by a whole cell is wrong.
        let mut offset = -spacing / 2.0;
        while offset + 9.0 * spacing <= n + spacing / 2.0 {
            let lines = (0..10)
                .map(|k| at(offset + k as f32 * spacing))
                .sum::<f32>()
                / 10.0;
            let outside = at(offset - spacing).max(at(offset + 10.0 * spacing));
            let score = lines - outside;
            if score > best.2 {
                best = (offset, spacing, score);
            }
            offset += 1.0;
        }
        spacing += 0.25;
    }
    best
}

/// Looks for the lattice of grid lines inside the rectified quadrilateral and
/// returns its corners. This recovers grids whose border is merged with a frame or
/// the edge of the paper, and corrects slightly misplaced corners.
fn refine(gray: &GrayImage, quad: &Quad) -> Option<Quad> {
    let s = SIDE as f32;
    let projection =
        Projection::from_control_points(*quad, [(0.0, 0.0), (s, 0.0), (s, s), (0.0, s)])?;
    let bin = binarize(&rectify(gray, quad)?, CELL / 2, 5);
    let mut rows = vec![0.0; SIDE as usize];
    let mut columns = vec![0.0; SIDE as usize];
    for (x, y) in (0..SIDE).flat_map(|y| (0..SIDE).map(move |x| (x, y))) {
        if thin_ink(&bin, x, y, true) {
            rows[y as usize] += 1.0 / s;
        }
        if thin_ink(&bin, x, y, false) {
            columns[x as usize] += 1.0 / s;
        }
    }
    let (x0, sx, _) = fit_lattice(&columns);
    let (y0, sy, _) = fit_lattice(&rows);
    let (x1, y1) = (x0 + 9.0 * sx, y0 + 9.0 * sy);
    let back = projection.invert();
    Some([(x0, y0), (x1, y0), (x1, y1), (x0, y1)].map(|p| back * p))
}

/// Labels of the connected components of a binary image.
type Labels = image::ImageBuffer<Luma<u32>, Vec<u32>>;

/// Intersection of the lines (a, b) and (c, d).
fn intersect(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> Option<(f64, f64)> {
    let (r, s) = ((b.0 - a.0, b.1 - a.1), (d.0 - c.0, d.1 - c.1));
    let det = r.0 * s.1 - r.1 * s.0;
    if det.abs() < 1e-9 {
        return None;
    }
    let t = ((c.0 - a.0) * s.1 - (c.1 - a.1) * s.0) / det;
    Some((a.0 + t * r.0, a.1 + t * r.1))
}

/// Quadrilateral supported by the four longest sides of the convex hull of a blob.
/// Unlike the extreme points, it finds the corners of grids without outer border,
/// whose hull has its corners cut off.
fn hull_quad(labels: &Labels, label: u32, blob: &Blob) -> Option<Quad> {
    let mut points = Vec::new();
    for y in blob.y0..=blob.y1 {
        let row: Vec<u32> = (blob.x0..=blob.x1)
            .filter(|&x| labels.get_pixel(x, y).0[0] == label)
            .collect();
        if let (Some(&first), Some(&last)) = (row.first(), row.last()) {
            points.push(Point::new(first as i32, y as i32));
            points.push(Point::new(last as i32, y as i32));
        }
    }
    let hull = convex_hull(points);
    let polygon = approximate_polygon_dp(&hull, arc_length(&hull, true) * 0.01, true);
    if polygon.len() < 4 {
        return None;
    }
    let n = polygon.len();
    let length = |i: usize| {
        let (a, b) = (polygon[i], polygon[(i + 1) % n]);
        (((b.x - a.x).pow(2) + (b.y - a.y).pow(2)) as f64).sqrt()
    };
    let mut sides: Vec<usize> = (0..n).collect();
    sides.sort_by(|&a, &b| length(b).total_cmp(&length(a)));
    sides.truncate(4);
    sides.sort();

    let point = |i: usize| (polygon[i % n].x as f64, polygon[i % n].y as f64);
    let mut corners = Vec::new();
    for k in 0..4 {
        let (i, j) = (sides[k], sides[(k + 1) % 4]);
        corners.push(intersect(point(i), point(i + 1), point(j), point(j + 1))?);
    }
    let pick = |key: fn(&(f64, f64)) -> f64| {
        *corners
            .iter()
            .max_by(|a, b| key(a).total_cmp(&key(b)))
            .unwrap()
    };
    let quad = [
        pick(|p| -p.0 - p.1),
        pick(|p| p.0 - p.1),
        pick(|p| p.0 + p.1),
        pick(|p| p.1 - p.0),
    ];
    Some(quad.map(|(x, y)| (x as f32, y as f32)))
}

/// Finds every sudoku grid of the picture, in reading order.
pub fn find_grids(gray: &GrayImage) -> Vec<Quad> {
    let (w, h) = gray.dimensions();
    let scale = (WORK_SIZE as f32 / w.max(h) as f32).min(1.0);
    let work = if scale < 1.0 {
        imageops::resize(
            gray,
            (w as f32 * scale) as u32,
            (h as f32 * scale) as u32,
            FilterType::Triangle,
        )
    } else {
        gray.clone()
    };
    let (ww, wh) = work.dimensions();
    let bin = binarize(&gaussian_blur_f32(&work, 1.0), (ww.max(wh) / 40).max(5), 10);
    let (blobs, labels) = blobs(&bin);

    let min_side = ww.min(wh) as f32 * 0.15;
    let mut candidates: Vec<(u32, Blob)> = blobs
        .into_iter()
        .enumerate()
        .map(|(i, b)| (i as u32 + 1, b))
        .filter(|(_, b)| {
            let (bw, bh) = (b.width() as f32, b.height() as f32);
            bw >= min_side && bh >= min_side && (0.75..=1.33).contains(&(bw / bh))
        })
        .collect();
    candidates.sort_by_key(|(_, b)| std::cmp::Reverse(b.width() * b.height()));

    let square = |q: &Quad| {
        let side =
            |a: (f32, f32), b: (f32, f32)| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        let ratio = (side(q[0], q[1]) + side(q[3], q[2])) / (side(q[0], q[3]) + side(q[1], q[2]));
        (0.7..=1.4).contains(&ratio)
    };
    let area = |q: &Quad| {
        let cross = |a: (f32, f32), b: (f32, f32), c: (f32, f32)| {
            (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
        };
        (cross(q[0], q[1], q[2]) + cross(q[0], q[2], q[3])).abs() / 2.0
    };
    let mut scored: Vec<(f32, Quad)> = Vec::new();
    for (label, blob) in candidates {
        let quads: Vec<Quad> = [
            Some(blob.extremes.map(|(x, y)| (x as f32, y as f32))),
            hull_quad(&labels, label, &blob),
        ]
        .iter()
        .flatten()
        .map(|quad| quad.map(|(x, y)| (x / scale, y / scale)))
        .flat_map(|quad| {
            // Refining twice converges when the first guess is far from the grid.
            let once = refine(gray, &quad);
            let twice = once.and_then(|q| refine(gray, &q));
            [Some(quad), once, twice]
        })
        .flatten()
        .filter(square)
        .collect();
        let scored_quads: Vec<(f32, Quad)> = quads
            .into_iter()
            .filter_map(|quad| {
                let rectified = rectify(gray, &quad)?;
                Some((line_score(&binarize(&rectified, CELL / 2, 5)), quad))
            })
            .collect();
        // A lattice of a part of the grid can score well too: the grid is the largest
        // of the good ones.
        let top = scored_quads.iter().map(|q| q.0).fold(f32::MIN, f32::max);
        let best = scored_quads
            .into_iter()
            .filter(|q| q.0 >= top * 0.75)
            .max_by(|a, b| area(&a.1).total_cmp(&area(&b.1)));
        if let Some(best) = best.filter(|b| b.0 >= MIN_LINE_SCORE) {
            scored.push(best);
        }
    }

    // Best grids first; a grid overlapping a better one is a duplicate.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0));
    let bounds = |q: &Quad| {
        let xs = q.map(|p| p.0);
        let ys = q.map(|p| p.1);
        let min = |v: [f32; 4]| v.iter().cloned().fold(f32::MAX, f32::min);
        let max = |v: [f32; 4]| v.iter().cloned().fold(f32::MIN, f32::max);
        (min(xs), min(ys), max(xs), max(ys))
    };
    let mut grids: Vec<((f32, f32, f32, f32), Quad)> = Vec::new();
    for (_, quad) in scored {
        let b = bounds(&quad);
        let overlaps = grids
            .iter()
            .any(|(g, _)| b.0 < g.2 && g.0 < b.2 && b.1 < g.3 && g.1 < b.3);
        if !overlaps {
            grids.push((b, quad));
        }
    }

    // Reading order: rows of grids from top to bottom, then left to right.
    let row_height = grids
        .first()
        .map_or(1.0, |(g, _)| (g.3 - g.1) / 2.0)
        .max(1.0);
    grids.sort_by_key(|(g, _)| ((g.1 / row_height) as u32, g.0 as u32));
    grids.into_iter().map(|(_, q)| q).collect()
}
