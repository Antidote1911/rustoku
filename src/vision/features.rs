//! Turns an isolated digit into a fixed-size feature vector.
//!
//! This file is shared with `examples/gen_digits.rs` (through `#[path]`), so that
//! training digits and digits read in pictures go through exactly the same
//! normalization.

use image::{imageops, imageops::FilterType, GrayImage, Luma};

/// Side of the square canvas a digit is drawn on.
pub const CANVAS: usize = 24;
/// Size of the longest side of the digit inside the canvas.
const BOX: f32 = 20.0;

pub type Normalized = [u8; CANVAS * CANVAS];

/// Scales a binary mask (255 = ink), already cropped to the digit bounding box,
/// so that its longest side is `BOX` pixels, and centers it on the canvas.
pub fn normalize(mask: &GrayImage) -> Normalized {
    let (w, h) = mask.dimensions();
    let scale = BOX / w.max(h) as f32;
    let tw = ((w as f32 * scale).round() as u32).max(1);
    let th = ((h as f32 * scale).round() as u32).max(1);
    let small = imageops::resize(mask, tw, th, FilterType::Triangle);

    let mut out = [0u8; CANVAS * CANVAS];
    let ox = (CANVAS as u32 - tw) / 2;
    let oy = (CANVAS as u32 - th) / 2;
    for (x, y, Luma([v])) in small.enumerate_pixels() {
        out[((y + oy) as usize) * CANVAS + (x + ox) as usize] = *v;
    }
    out
}

/// Number of holes of a binary mask (255 = ink), such as the loops of 6, 8 or 9.
/// Tiny holes, due to noise, are ignored.
pub fn holes(mask: &GrayImage) -> u8 {
    let (w, h) = (mask.width() as i32 + 2, mask.height() as i32 + 2);
    let min_area = ((w * h) / 200).max(3);
    let background = |x: i32, y: i32| {
        x <= 0
            || y <= 0
            || x >= w - 1
            || y >= h - 1
            || mask.get_pixel(x as u32 - 1, y as u32 - 1).0[0] < 128
    };
    let mut seen = vec![false; (w * h) as usize];
    let mut count = 0;
    for start in 0..w * h {
        let (sx, sy) = (start % w, start / w);
        if seen[start as usize] || !background(sx, sy) {
            continue;
        }
        let (mut area, mut outer) = (0, false);
        let mut stack = vec![(sx, sy)];
        seen[start as usize] = true;
        while let Some((x, y)) = stack.pop() {
            area += 1;
            outer |= x == 0 || y == 0 || x == w - 1 || y == h - 1;
            for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if nx >= 0
                    && ny >= 0
                    && nx < w
                    && ny < h
                    && !seen[(ny * w + nx) as usize]
                    && background(nx, ny)
                {
                    seen[(ny * w + nx) as usize] = true;
                    stack.push((nx, ny));
                }
            }
        }
        if !outer && area >= min_area {
            count += 1;
        }
    }
    count
}
