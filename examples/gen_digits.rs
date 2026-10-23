//! Builds the training set of the digit classifier of `rustoku photo`.
//!
//! Font files are read from stdin, one path per line:
//!
//! ```bash
//! fc-list -f '%{file}\n' | cargo run --release --example gen_digits -- digits.dat
//! python3 tools/train_digits.py digits.dat src/vision/digits.bin
//! ```
//!
//! Every digit of every font is rendered many times with the degradations of a
//! photo (small size, blur, rotation, stroke weight), then goes through the same
//! normalization as the digits read by `rustoku`. Symbol or decorative fonts,
//! whose digits do not look like the digits of the other fonts, are skipped.
//!
//! Record format: digit (u8), holes (u8), font index (u16 LE), normalized image.

extern crate ab_glyph;
extern crate image;
extern crate imageproc;

#[path = "../src/vision/features.rs"]
mod features;

use std::{
    collections::BTreeSet,
    io::{self, BufRead, Write},
};

use ab_glyph::{Font, FontVec, PxScale};
use features::{holes, normalize, Normalized, CANVAS};
use image::{imageops, imageops::FilterType, GrayImage, Luma};
use imageproc::{
    filter::gaussian_blur_f32,
    geometric_transformations::{warp_into, Border, Interpolation, Projection},
};

const SAMPLES_PER_DIGIT: usize = 20;
/// Digit height in the rectified grid of `rustoku photo` (half a cell).
const TARGET_HEIGHT: u32 = 34;

/// xorshift64*: good enough to draw augmentation parameters.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, low: f32, high: f32) -> f32 {
        low + (high - low) * self.next()
    }
}

/// Anti-aliased rendering of a digit, `px` pixels per em, with a margin.
fn render(font: &FontVec, digit: char, px: f32) -> Option<GrayImage> {
    let id = font.glyph_id(digit);
    if id.0 == 0 {
        return None;
    }
    let outline = font.outline_glyph(id.with_scale(PxScale::from(px)))?;
    let bounds = outline.px_bounds();
    let pad = 4;
    let (w, h) = (
        bounds.width() as u32 + 2 * pad,
        bounds.height() as u32 + 2 * pad,
    );
    if bounds.height() < 4.0 {
        return None;
    }
    let mut img = GrayImage::new(w, h);
    outline.draw(|x, y, c| img.put_pixel(x + pad, y + pad, Luma([(c * 255.0) as u8])));
    Some(img)
}

fn crop_to_ink(img: &GrayImage) -> Option<GrayImage> {
    let (mut x0, mut y0, mut x1, mut y1) = (u32::MAX, u32::MAX, 0, 0);
    for (x, y, Luma([v])) in img.enumerate_pixels() {
        if *v > 127 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    (x0 <= x1).then(|| imageops::crop_imm(img, x0, y0, x1 - x0 + 1, y1 - y0 + 1).to_image())
}

/// A degraded binary version of the digit, as `rustoku photo` would extract it:
/// rendered small then scaled to the size of digits in the rectified grid,
/// stretched, slanted, rotated, blurred and thresholded at a random level
/// (which makes strokes thinner or bolder).
fn degrade(font: &FontVec, digit: char, rng: &mut Rng, clean: bool) -> Option<GrayImage> {
    let px = if clean { 96.0 } else { rng.range(14.0, 80.0) };
    let mut img = render(font, digit, px)?;
    let scale = TARGET_HEIGHT as f32 / (img.height() as f32 - 8.0).max(1.0);
    img = imageops::resize(
        &img,
        ((img.width() as f32 * scale) as u32).max(1),
        ((img.height() as f32 * scale) as u32).max(1),
        FilterType::Triangle,
    );
    let threshold = if clean {
        128.0
    } else {
        let (stretch, shear, angle) = (
            rng.range(0.85, 1.2),
            rng.range(-0.2, 0.15),
            rng.range(-0.06, 0.06),
        );
        let (w, h) = (img.width() as f32, img.height() as f32);
        let (cw, ch) = (w * 1.5 + 16.0, h + 16.0);
        // Centered stretch and shear, then rotation, onto a larger canvas.
        let center = Projection::translate(cw / 2.0, ch / 2.0);
        let affine = Projection::from_matrix([stretch, shear, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])?;
        let projection =
            center * Projection::rotate(angle) * affine * Projection::translate(-w / 2.0, -h / 2.0);
        let mut out = GrayImage::new(cw as u32, ch as u32);
        warp_into(
            &img,
            projection,
            Interpolation::Bilinear,
            Border::Constant(Luma([0])),
            &mut out,
        );
        img = out;
        let blur = rng.range(0.0, 1.4);
        if blur > 0.3 {
            img = gaussian_blur_f32(&img, blur);
        }
        rng.range(70.0, 185.0)
    };
    img.pixels_mut()
        .for_each(|p| p.0[0] = if p.0[0] as f32 > threshold { 255 } else { 0 });
    let mask = crop_to_ink(&img)?;
    // Too strong degradations break thin glyphs or fill bold ones: not a digit any more.
    let ink = mask.pixels().filter(|p| p.0[0] > 0).count() as f32;
    let full = (mask.width() * mask.height()) as f32;
    (clean || (mask.height() * 10 >= TARGET_HEIGHT * 7 && ink < 0.75 * full)).then_some(mask)
}

/// Zero-mean, unit-norm image, to detect fonts with unusual digits by correlation.
fn centered(n: &Normalized) -> Vec<f32> {
    let mean = n.iter().map(|&v| v as f32).sum::<f32>() / n.len() as f32;
    let v: Vec<f32> = n.iter().map(|&v| v as f32 - mean).collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.into_iter().map(|x| x / norm).collect()
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .expect("usage: gen_digits <output> < font list");
    let paths: BTreeSet<String> = io::stdin()
        .lock()
        .lines()
        .map_while(Result::ok)
        .filter(|l| {
            let l = l.to_lowercase();
            l.ends_with(".ttf") || l.ends_with(".otf") || l.ends_with(".ttc")
        })
        .collect();

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let fonts: Vec<FontVec> = paths
        .iter()
        .filter_map(|p| FontVec::try_from_vec_and_index(std::fs::read(p).ok()?, 0).ok())
        .collect();
    let clean: Vec<Option<Vec<Vec<f32>>>> = fonts
        .iter()
        .map(|font| {
            (1..=9u8)
                .map(|d| {
                    degrade(font, char::from(b'0' + d), &mut rng, true)
                        .map(|m| centered(&normalize(&m)))
                })
                .collect()
        })
        .collect();

    // A font is kept when the other fonts recognize at least 8 of its digits.
    let kept: Vec<usize> = (0..fonts.len())
        .filter(|&i| {
            let Some(digits) = &clean[i] else {
                return false;
            };
            let correct = digits
                .iter()
                .enumerate()
                .filter(|(d, img)| {
                    let mut best = (f32::MIN, 0);
                    for other in clean
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .filter_map(|(_, c)| c.as_ref())
                    {
                        for (e, o) in other.iter().enumerate() {
                            let s: f32 = img.iter().zip(o).map(|(x, y)| x * y).sum();
                            if s > best.0 {
                                best = (s, e);
                            }
                        }
                    }
                    best.1 == *d
                })
                .count();
            correct >= 8
        })
        .collect();
    eprintln!(
        "{} font files, {} with digits 1-9, {} kept",
        paths.len(),
        clean.iter().flatten().count(),
        kept.len()
    );

    let mut file =
        io::BufWriter::new(std::fs::File::create(&out_path).expect("cannot create output"));
    let mut count = 0;
    for (index, &i) in kept.iter().enumerate() {
        for d in 1..=9u8 {
            for _ in 0..SAMPLES_PER_DIGIT {
                if let Some(mask) = degrade(&fonts[i], char::from(b'0' + d), &mut rng, false) {
                    file.write_all(&[d, holes(&mask)]).unwrap();
                    file.write_all(&(index as u16).to_le_bytes()).unwrap();
                    file.write_all(&normalize(&mask)).unwrap();
                    count += 1;
                }
            }
        }
    }
    eprintln!("{} samples of {} bytes", count, 4 + CANVAS * CANVAS);
}
