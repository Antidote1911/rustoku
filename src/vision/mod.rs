//! Reading sudokus from pictures.
//!
//! 1. `detect` finds the grids (largest square connected components whose
//!    inner lines are present) and rectifies their perspective.
//! 2. Each cell is binarized; the most central blob of digit size is the digit,
//!    a cell without one is empty.
//! 3. Digits are recognized by a small neural network trained on digits of many
//!    fonts, rendered with the degradations of a photo (`digits.bin`, built by
//!    `examples/gen_digits.rs` and `tools/train_digits.py`).
//! 4. If the read grid has no solution, the least confident digits are replaced
//!    by their next best guesses until the solver finds one.

mod detect;
mod features;

use std::{convert::TryInto, path::Path, sync::OnceLock};

use image::{imageops, GrayImage, Luma};
use imageproc::contrast::otsu_level;
use sudoku::Sudoku;

use self::detect::{binarize, blobs, CELL};

/// Log-probability of each digit 1..=9 for a cell, or `None` for an empty cell.
pub type CellScores = Option<[f32; 9]>;

pub struct Reading {
    pub cells: Vec<CellScores>,
}

/// Cell margin ignored to get rid of the grid lines.
const MARGIN: u32 = CELL / 12;

/// Weights of the digit classifier, built by `tools/train_digits.py`.
static MODEL: &[u8] = include_bytes!("digits.bin");

/// Multilayer perceptron: normalized image + number of holes -> hidden ReLU layer -> 9 digits.
struct Model {
    inputs: usize,
    hidden: usize,
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
}

fn model() -> &'static Model {
    static PARSED: OnceLock<Model> = OnceLock::new();
    PARSED.get_or_init(|| {
        assert_eq!(&MODEL[..4], b"RTKM", "invalid digits.bin");
        let u32_at = |i: usize| u32::from_le_bytes(MODEL[i..i + 4].try_into().unwrap()) as usize;
        let (inputs, hidden, outputs) = (u32_at(4), u32_at(8), u32_at(12));
        assert_eq!(outputs, 9);
        let mut floats = MODEL[16..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c));
        let mut take = |n: usize| floats.by_ref().take(n).collect::<Vec<f32>>();
        Model {
            inputs,
            hidden,
            w1: take(inputs * hidden),
            b1: take(hidden),
            w2: take(hidden * 9),
            b2: take(9),
        }
    })
}

/// Log-probabilities of the digits 1..=9 for a digit mask.
fn classify(mask: &GrayImage) -> [f32; 9] {
    let m = model();
    let mut input: Vec<f32> = features::normalize(mask)
        .iter()
        .map(|&v| v as f32 / 255.0)
        .collect();
    let holes = features::holes(mask).min(2) as usize;
    input.extend((0..3).map(|h| if h == holes { 1.0 } else { 0.0 }));
    debug_assert_eq!(input.len(), m.inputs);

    let mut hidden = m.b1.clone();
    for (i, &x) in input.iter().enumerate().filter(|(_, &x)| x != 0.0) {
        let row = &m.w1[i * m.hidden..(i + 1) * m.hidden];
        hidden.iter_mut().zip(row).for_each(|(h, w)| *h += x * w);
    }
    let mut logits = [0.0f32; 9];
    for (j, l) in logits.iter_mut().enumerate() {
        *l = m.b2[j]
            + (0..m.hidden)
                .map(|k| hidden[k].max(0.0) * m.w2[k * 9 + j])
                .sum::<f32>();
    }
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let log_sum = logits.iter().map(|l| (l - max).exp()).sum::<f32>().ln() + max;
    logits.map(|l| l - log_sum)
}

/// Mask of the digit of a cell, if any. `cell` is a binarized cell, 255 = ink.
fn extract_digit(cell: &GrayImage) -> Option<GrayImage> {
    let n = cell.width();
    let (blobs, labels) = blobs(cell);
    let center = n as f32 / 2.0;
    let (index, blob) = blobs
        .iter()
        .enumerate()
        .filter(|(_, b)| {
            let (cx, cy) = ((b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0);
            b.height() >= CELL * 28 / 100
                && b.height() < n - 2
                && b.width() < n - 2
                && b.width() <= CELL * 3 / 4
                && b.area >= CELL * CELL / 150
                && (cx - center).abs() <= CELL as f32 * 0.3
                && (cy - center).abs() <= CELL as f32 * 0.3
        })
        .max_by_key(|(_, b)| b.area)?;
    // Glyphs can be broken in several pieces (thin strokes, compression artifacts):
    // keep the pieces close to the main one that do not touch the cell border.
    let pad = CELL / 8;
    let pieces: Vec<(u32, &detect::Blob)> = blobs
        .iter()
        .enumerate()
        .filter(|&(i, b)| {
            i == index
                || (b.x0 > 0
                    && b.y0 > 0
                    && b.x1 < n - 1
                    && b.y1 < n - 1
                    && b.x1 + pad >= blob.x0
                    && b.x0 <= blob.x1 + pad
                    && b.y1 + pad >= blob.y0
                    && b.y0 <= blob.y1 + pad)
        })
        .map(|(i, b)| (i as u32 + 1, b))
        .collect();
    let x0 = pieces.iter().map(|(_, b)| b.x0).min()?;
    let y0 = pieces.iter().map(|(_, b)| b.y0).min()?;
    let x1 = pieces.iter().map(|(_, b)| b.x1).max()?;
    let y1 = pieces.iter().map(|(_, b)| b.y1).max()?;
    let mut mask = GrayImage::new(x1 - x0 + 1, y1 - y0 + 1);
    for (x, y, p) in mask.enumerate_pixels_mut() {
        let label = labels.get_pixel(x0 + x, y0 + y).0[0];
        if pieces.iter().any(|(l, _)| *l == label) {
            *p = Luma([255]);
        }
    }
    Some(mask)
}

/// Reads every sudoku grid of a picture. When `debug` is set, the rectified
/// and binarized grids are saved there.
pub fn read_picture(path: &Path, debug: Option<&Path>) -> Result<Vec<Reading>, String> {
    let gray = image::open(path).map_err(|e| e.to_string())?.to_luma8();
    let mut readings = Vec::new();
    for (i, quad) in detect::find_grids(&gray).iter().enumerate() {
        let Some(grid) = detect::rectify(&gray, quad) else {
            continue;
        };
        let bin = binarize(&grid, CELL / 2, 15);
        if let Some(dir) = debug {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy();
            let save = |img: &GrayImage, kind: &str| {
                let _ = img.save(dir.join(format!("{}_{}_{}.png", stem, i + 1, kind)));
            };
            save(&grid, "grid");
            save(&bin, "bin");
        }
        let cells = (0..81)
            .map(|i| {
                let (x, y) = (
                    (i % 9) as u32 * CELL + MARGIN,
                    (i / 9) as u32 * CELL + MARGIN,
                );
                let size = CELL - 2 * MARGIN;
                let mut cell = imageops::crop_imm(&bin, x, y, size, size).to_image();
                // The local threshold also takes the blurry halo of the strokes as ink: the
                // Otsu level of the cell, halfway between ink and paper, trims it.
                let gray = imageops::crop_imm(&grid, x, y, size, size).to_image();
                let level = otsu_level(&gray);
                for (p, g) in cell.pixels_mut().zip(gray.pixels()) {
                    if g.0[0] > level {
                        p.0[0] = 0;
                    }
                }
                extract_digit(&cell).map(|mask| classify(&mask))
            })
            .collect();
        readings.push(Reading { cells });
    }
    Ok(readings)
}

pub enum Status {
    Unique(Sudoku),
    Multiple(Sudoku),
    Unsolvable,
}

pub struct Resolution {
    /// The grid as read, after corrections.
    pub grid: Sudoku,
    /// (cell index, digit read, digit kept; 0 = empty)
    pub corrections: Vec<(usize, u8, u8)>,
    pub status: Status,
}

/// Pairs of cells holding the same digit in a row, a column or a box.
pub fn conflicts(grid: &Sudoku) -> Vec<(usize, usize)> {
    let bytes = grid.to_bytes();
    let mut pairs = Vec::new();
    for a in 0..81 {
        for b in a + 1..81 {
            let (ra, ca, rb, cb) = (a / 9, a % 9, b / 9, b % 9);
            let same_unit = ra == rb || ca == cb || (ra / 3 == rb / 3 && ca / 3 == cb / 3);
            if bytes[a] != 0 && bytes[a] == bytes[b] && same_unit {
                pairs.push((a, b));
            }
        }
    }
    pairs
}

fn ranked(scores: &[f32; 9]) -> Vec<(u8, f32)> {
    let mut r: Vec<(u8, f32)> = scores
        .iter()
        .enumerate()
        .map(|(d, &s)| (d as u8 + 1, s))
        .collect();
    r.sort_by(|a, b| b.1.total_cmp(&a.1));
    r
}

fn status_of(bytes: &[u8; 81]) -> (Sudoku, Status) {
    let sudoku = Sudoku::from_bytes(*bytes).expect("digits are 0..=9");
    let status = match sudoku.solutions_up_to(2).as_slice() {
        [] => Status::Unsolvable,
        [s] => Status::Unique(*s),
        [s, ..] => Status::Multiple(*s),
    };
    (sudoku, status)
}

/// Builds the grid from the best guesses. If it has no solution, tries to replace
/// one or two of the least confident digits by their second or third guess, or
/// to drop them, cheapest change first.
pub fn resolve(reading: &Reading) -> Resolution {
    const SUSPECTS: usize = 12;
    // Cost of dropping a digit, compared to the log-probability lost by a substitution.
    const DROP_COST: f32 = 2.0;
    // Beyond this, the digits were read with confidence: if the grid has no
    // solution, the puzzle itself is probably invalid.
    const MAX_COST: f32 = 4.0;

    let mut base = [0u8; 81];
    // (confidence margin of the cell, cell, alternative digit, cost of the alternative)
    let mut alternatives: Vec<(f32, usize, u8, f32)> = Vec::new();
    for (i, scores) in reading.cells.iter().enumerate() {
        if let Some(scores) = scores {
            let r = ranked(scores);
            base[i] = r[0].0;
            let margin = r[0].1 - r[1].1;
            alternatives.extend(r[1..3].iter().map(|&(d, s)| (margin, i, d, r[0].1 - s)));
            alternatives.push((margin, i, 0, DROP_COST + margin));
        }
    }
    let (grid, status) = status_of(&base);
    if !matches!(status, Status::Unsolvable) {
        return Resolution {
            grid,
            corrections: Vec::new(),
            status,
        };
    }

    // The 3 alternatives of the least confident cells.
    alternatives.sort_by(|a, b| a.0.total_cmp(&b.0));
    alternatives.truncate(3 * SUSPECTS);
    let singles: Vec<(usize, u8, f32)> =
        alternatives.iter().map(|&(_, i, d, c)| (i, d, c)).collect();
    let mut changes: Vec<(f32, Vec<(usize, u8)>)> =
        singles.iter().map(|&(i, d, c)| (c, vec![(i, d)])).collect();
    for (a, &(i, d, c)) in singles.iter().enumerate() {
        for &(j, e, k) in &singles[a + 1..] {
            if i != j {
                changes.push((c + k, vec![(i, d), (j, e)]));
            }
        }
    }
    changes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut fallback = None;
    for (_, change) in changes.iter().take_while(|(cost, _)| *cost <= MAX_COST) {
        let mut bytes = base;
        change.iter().for_each(|&(i, d)| bytes[i] = d);
        let (grid, status) = status_of(&bytes);
        let corrections = change.iter().map(|&(i, d)| (i, base[i], d)).collect();
        match status {
            Status::Unique(_) => {
                return Resolution {
                    grid,
                    corrections,
                    status,
                }
            }
            Status::Multiple(_) if fallback.is_none() => {
                fallback = Some(Resolution {
                    grid,
                    corrections,
                    status,
                })
            }
            _ => {}
        }
    }
    fallback.unwrap_or(Resolution {
        grid,
        corrections: Vec::new(),
        status: Status::Unsolvable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compares the recognition of the pictures of `grilles/` with the
    /// hand-made transcriptions of `grilles/expected.txt` (one grid per picture,
    /// or several in reading order).
    #[test]
    fn pictures_are_read_correctly() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("grilles");
        let expected = std::fs::read_to_string(dir.join("expected.txt")).unwrap();
        let mut failures = Vec::new();
        for line in expected.lines().filter(|l| !l.starts_with('#')) {
            // File names may contain spaces: the grids are the trailing 81-character words.
            let words: Vec<&str> = line.split(' ').collect();
            let count = words.iter().rev().take_while(|w| w.len() == 81).count();
            let file = words[..words.len() - count].join(" ");
            let grids = &words[words.len() - count..];
            let read: Vec<String> = read_picture(&dir.join(&file), None)
                .unwrap()
                .iter()
                .map(|r| resolve(r).grid.to_str_line().to_string())
                .collect();
            if read != grids {
                failures.push(format!(
                    "{}\n  expected {:?}\n  read     {:?}",
                    file, grids, read
                ));
            }
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}
