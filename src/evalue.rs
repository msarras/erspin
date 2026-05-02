//! E-value computation by histogram convolution, ported from C ERPIN
//! (`libsrc/Eval.c`, `cdf.c`, `mhisto.c`, `hshisto.c`, `histools.c`,
//! `conv.c`).
//!
//! The algorithm builds a discrete-score histogram for the mask under random
//! background, takes its right-tail CDF, applies extreme-value correction
//! `evprob(p, ncfg) = 1 - (1-p)^N`, and scales by database volume to get the
//! expected number of hits at-or-above each score.
//!
//! The Rust port mirrors C's representation: histogram bins are evenly-
//! spaced grid points with width `dh = (hmax - hmin) / (bins - 1)`, and
//! after every convolution step the histogram is renormalized to integral
//! 1. Skipping renormalization between columns / elements (as the previous
//! Rust implementation did) compounds a per-step scaling error and produced
//! E-values ~10^13 too small.

use crate::profile::{Background, LOG_ZERO};
use crate::types::*;

/// Bin width for E-value histograms. Matches C's `DELTA_H` in `rnaIV.h`.
const DELTA_H: f64 = 0.05;

/// Score threshold below which we treat a profile cell as "missing" (a
/// LOG_ZERO floor). Matches C's `0.8 * LOG_ZERO` in `hshisto.c`.
const FINITE_THRESHOLD: f64 = 0.8 * LOG_ZERO;

/// 5-point margin (`Margin = 2.5` in `histools.c::SetupHist`) added on each
/// side of the input score range.
const SETUP_MARGIN: f64 = 2.5;

/// Discrete histogram with uniform-spaced grid points. `vals[i]` is the
/// histogram value at score `hmin + i * dh`, where `dh = (hmax - hmin) /
/// (bins - 1)`. Integral (sum × dh) is normalized to 1 after `normalize`.
#[derive(Debug, Clone)]
struct Histo {
    vals: Vec<f64>,
    hmin: f64,
    hmax: f64,
}

impl Histo {
    fn bins(&self) -> usize {
        self.vals.len()
    }

    fn dh(&self) -> f64 {
        if self.bins() < 2 {
            DELTA_H
        } else {
            (self.hmax - self.hmin) / (self.bins() - 1) as f64
        }
    }

    /// Map a continuous score to bin index, mirroring C's `GetX`.
    fn bin_index(&self, score: f64) -> usize {
        let dh = self.dh();
        let raw = ((score - self.hmin) / dh).round() as isize;
        raw.clamp(0, self.bins() as isize - 1) as usize
    }

    /// Sum of vals × dh. Should be ~1 after `normalize`.
    fn integral(&self) -> f64 {
        self.vals.iter().sum::<f64>() * self.dh()
    }

    /// Normalize so integral == 1 (mirrors C's `NormalizeHist`).
    fn normalize(&mut self) {
        let s = self.integral();
        if s > 0.0 {
            for v in &mut self.vals {
                *v /= s;
            }
        }
    }
}

/// Mirror C's `SetupHist`: build an empty histogram covering [min, max]
/// with bin width `dh`, plus a 2.5-bin margin on each side.
fn setup_hist(min: f64, max: f64, dh: f64) -> Histo {
    debug_assert!(min <= max, "setup_hist: min > max ({} > {})", min, max);
    debug_assert!(dh > 0.0, "setup_hist: dh must be > 0");

    let hmin = min - SETUP_MARGIN * dh;
    let hmax_raw = max + SETUP_MARGIN * dh;
    let bins = 1 + ((hmax_raw - hmin) / dh).ceil() as usize;
    let hmax = hmin + (bins - 1) as f64 * dh;

    Histo {
        vals: vec![0.0; bins],
        hmin,
        hmax,
    }
}

/// Build a histogram from weighted samples. Mirrors C's `GetNHistW`.
fn weighted_histogram(scores: &[f64], weights: &[f64], dh: f64) -> Histo {
    debug_assert_eq!(scores.len(), weights.len());
    let mut min = scores[0];
    let mut max = scores[0];
    for &s in scores.iter().skip(1) {
        if s < min {
            min = s;
        }
        if s > max {
            max = s;
        }
    }

    let mut hist = setup_hist(min, max, dh);
    let mut total = 0.0;
    for (&s, &w) in scores.iter().zip(weights) {
        let idx = hist.bin_index(s);
        hist.vals[idx] += w;
        total += w;
    }

    // Normalize so integral == 1 ⇒ vals[i] /= (total * dh).
    if total > 0.0 {
        let denom = total * dh;
        for v in &mut hist.vals {
            *v /= denom;
        }
    }
    hist
}

/// Direct discrete convolution. Mirrors C's `convlv` + `ConvHist`. After
/// the convolution `out.bins = h1.bins + h2.bins - 1`, and the support
/// shifts: `out.hmin = h1.hmin + h2.hmin`, `out.hmax = h1.hmax + h2.hmax`.
/// Both inputs must share the same bin width.
fn convolve(h1: &Histo, h2: &Histo) -> Histo {
    let n1 = h1.bins();
    let n2 = h2.bins();
    let n_out = n1 + n2 - 1;
    let mut vals = vec![0.0f64; n_out];

    for i in 0..n1 {
        let v1 = h1.vals[i];
        if v1 == 0.0 {
            continue;
        }
        for j in 0..n2 {
            vals[i + j] += v1 * h2.vals[j];
        }
    }

    Histo {
        vals,
        hmin: h1.hmin + h2.hmin,
        hmax: h1.hmax + h2.hmax,
    }
}

/// Convolve `column` into `running` and renormalize. Mirrors the inner
/// pattern in `hshisto.c` (e.g. `StsNoGHist:213`) where every column
/// histogram is convolved into the running total *and the running total
/// is renormalized after each step*. Skipping the normalization compounds
/// scaling errors over many columns.
fn convolve_into(running: &mut Histo, column: &Histo) {
    *running = convolve(running, column);
    running.normalize();
}

/// Build a single column histogram for a no-gap strand profile column.
/// Mirrors `hshisto.c::GetStNoGHist` inner loop: ATGC nucleotides only,
/// weighted by background frequency, skipping codes whose profile value
/// is at the `LOG_ZERO` floor. Returns `(hist, prob_finite)` where
/// `prob_finite` is the sum of background frequencies of finite codes.
fn strand_column_hist_nogap(profile: &[Vec<f64>], col: usize, bg: &Background) -> (Histo, f64) {
    let mut scores = Vec::with_capacity(4);
    let mut weights = Vec::with_capacity(4);
    let mut prob_finite = 0.0;

    for nt in 0..4 {
        let v = profile[nt][col];
        if v > FINITE_THRESHOLD {
            prob_finite += bg.freq[nt];
            scores.push(v);
            weights.push(bg.freq[nt]);
        }
    }

    if scores.is_empty() {
        // All-zero column: degenerate single-bin histogram at 0.
        return (
            Histo {
                vals: vec![1.0 / DELTA_H],
                hmin: 0.0,
                hmax: 0.0,
            },
            prob_finite,
        );
    }

    (weighted_histogram(&scores, &weights, DELTA_H), prob_finite)
}

/// Build a single column histogram for a gapped strand profile column.
/// Mirrors `hshisto.c::GStHist`: per ATGC, take `MAX(Prof[i][col], Prof[gap_row][col])`
/// (the larger of nucleotide and gap log-odds), weighted by full background.
/// C does not track a `Prfsc` factor for gapped strands.
fn strand_column_hist_gapped(profile: &[Vec<f64>], col: usize, bg: &Background) -> Histo {
    let mut scores = [0.0f64; 4];
    let mut weights = [0.0f64; 4];
    let gap_score = profile[4][col]; // gap row
    for nt in 0..4 {
        scores[nt] = profile[nt][col].max(gap_score);
        weights[nt] = bg.freq[nt];
    }
    weighted_histogram(&scores, &weights, DELTA_H)
}

/// Build a single column histogram for a helix profile column.
/// Mirrors `hshisto.c::GetSSHist`: 16 dinucleotide pairs (ATGC × ATGC),
/// weighted by `bg[5'] * bg[3']`, skipping pairs at the LOG_ZERO floor.
fn helix_column_hist(profile: &[Vec<f64>], col: usize, bg: &Background) -> (Histo, f64) {
    let mut scores = Vec::with_capacity(16);
    let mut weights = Vec::with_capacity(16);
    let mut prob_finite = 0.0;

    for b5 in 0..4 {
        for b3 in 0..4 {
            let code = b5 * 4 + b3;
            let v = profile[code][col];
            if v > FINITE_THRESHOLD {
                let w = bg.freq[b5] * bg.freq[b3];
                prob_finite += w;
                scores.push(v);
                weights.push(w);
            }
        }
    }

    if scores.is_empty() {
        return (
            Histo {
                vals: vec![1.0 / DELTA_H],
                hmin: 0.0,
                hmax: 0.0,
            },
            prob_finite,
        );
    }

    (weighted_histogram(&scores, &weights, DELTA_H), prob_finite)
}

/// Build the full strand element histogram by iterative column convolution.
/// For no-gap strands, mirrors `hshisto.c::GetStNoGHist`: tracks Prfsc.
/// For gapped strands, mirrors `hshisto.c::GStHist`: uses MAX(nt, gap)
/// approximation per column and does NOT track Prfsc (always 1.0).
fn strand_element_hist(strand: &Strand, bg: &Background) -> (Histo, f64) {
    let cols = strand.max_len;
    if strand.max_gaps == 0 {
        let (mut running, mut prfsc) = strand_column_hist_nogap(&strand.profile, 0, bg);
        for col in 1..cols {
            let (col_hist, p) = strand_column_hist_nogap(&strand.profile, col, bg);
            prfsc *= p;
            convolve_into(&mut running, &col_hist);
        }
        (running, prfsc)
    } else {
        let mut running = strand_column_hist_gapped(&strand.profile, 0, bg);
        for col in 1..cols {
            let col_hist = strand_column_hist_gapped(&strand.profile, col, bg);
            convolve_into(&mut running, &col_hist);
        }
        (running, 1.0)
    }
}

/// Build the full helix element histogram by iterative column convolution.
fn helix_element_hist(helix: &Helix, bg: &Background) -> (Histo, f64) {
    let cols = helix.helix_len;
    let (mut running, mut prfsc) = helix_column_hist(&helix.profile, 0, bg);
    for col in 1..cols {
        let (col_hist, p) = helix_column_hist(&helix.profile, col, bg);
        prfsc *= p;
        convolve_into(&mut running, &col_hist);
    }
    (running, prfsc)
}

/// Build the combined histogram for all elements in a mask. Mirrors
/// `mhisto.c::GetMaskHist`: convolve all helix columns into one histogram,
/// convolve all strand columns into another, then combine. We fold the two
/// branches together since the result is the same convolution either way.
fn mask_hist(pattern: &Pattern, mask: &ResolvedMask, bg: &Background) -> (Histo, f64) {
    let mut prfsc = 1.0;
    let mut running: Option<Histo> = None;

    for &si in &mask.st_indices {
        let (h, p) = strand_element_hist(&pattern.strands[si], bg);
        prfsc *= p;
        match running.as_mut() {
            None => running = Some(h),
            Some(r) => convolve_into(r, &h),
        }
    }
    for &hi in &mask.hx_indices {
        let (h, p) = helix_element_hist(&pattern.helices[hi], bg);
        prfsc *= p;
        match running.as_mut() {
            None => running = Some(h),
            Some(r) => convolve_into(r, &h),
        }
    }

    let mut hist = running.unwrap_or_else(|| Histo {
        vals: vec![1.0],
        hmin: 0.0,
        hmax: 0.0,
    });
    hist.normalize();
    (hist, prfsc)
}

/// Right-tail CDF, scaled by bin width. Mirrors `cdf.c::GetHistCdf`:
/// `cdf[i] = dh * sum(vals[j] for j ≥ i)`. For a normalized hist (integral
/// 1), `cdf[0]` should be 1.
fn right_tail_cdf(hist: &Histo) -> Histo {
    let n = hist.bins();
    let dh = hist.dh();
    let mut vals = vec![0.0f64; n];
    if n == 0 {
        return Histo {
            vals,
            hmin: hist.hmin,
            hmax: hist.hmax,
        };
    }
    vals[n - 1] = hist.vals[n - 1];
    for i in (0..n - 1).rev() {
        vals[i] = vals[i + 1] + hist.vals[i];
    }
    for v in &mut vals {
        *v *= dh;
    }
    Histo {
        vals,
        hmin: hist.hmin,
        hmax: hist.hmax,
    }
}

/// Extreme-value probability: `P(max of N IID trials > x | P(single > x) = p)`.
/// Mirrors `Eval.c::evprob`. Uses the binomial expansion for `p*N < 1` to
/// preserve precision when `1 - (1-p)^N` would round to 0.
fn evprob(mut p: f64, n: usize) -> f64 {
    if p >= 1.0 {
        return 1.0;
    }
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return p;
    }
    let n_f = n as f64;
    if p * n_f >= 1.0 {
        return 1.0 - (1.0 - p).powi(n as i32);
    }
    // Binomial expansion: A = N q + N(N-1) q^2 / 2 + ... with q = -p.
    p = -p;
    let max_order = 30usize;
    let mut a = 0.0f64;
    let mut b = 1.0f64;
    let limit = max_order.min(n);
    for i in 1..=limit {
        b *= (n_f - i as f64 + 1.0) * p / i as f64;
        a += b;
    }
    -a
}

/// Linear interpolation of a CDF histogram, with the C ERPIN
/// `-0.25 * dh` offset for sparse-bin smoothing (`cdf.c::Interpol`).
fn interpolate(cdf: &Histo, score: f64) -> f64 {
    let n = cdf.bins();
    if n == 0 {
        return 0.0;
    }
    let dh = cdf.dh();
    let x = score - 0.25 * dh;
    if x <= cdf.hmin {
        return cdf.vals[0];
    }
    if x >= cdf.hmax {
        return cdf.vals[n - 1];
    }
    let raw = (x - cdf.hmin) / dh;
    let idx = raw.floor() as usize;
    if idx + 1 >= n {
        return cdf.vals[n - 1];
    }
    let frac = raw - idx as f64;
    cdf.vals[idx] + frac * (cdf.vals[idx + 1] - cdf.vals[idx])
}

/// Precomputed E-value lookup table for a mask.
pub struct EvalueTable {
    cdf: Histo,
}

impl EvalueTable {
    /// Build E-value table for the given mask. `datavol` is the total
    /// nucleotide count scanned (sum of sequence lengths × strand passes).
    pub fn build(
        pattern: &Pattern,
        mask: &ResolvedMask,
        bg: &Background,
        datavol: f64,
    ) -> Self {
        let (hist, prfsc) = mask_hist(pattern, mask, bg);
        let mut cdf = right_tail_cdf(&hist);

        let ncfg = mask.configs.len().max(1);
        for v in &mut cdf.vals {
            let prob = prfsc * (*v);
            *v = datavol * evprob(prob, ncfg);
        }

        Self { cdf }
    }

    /// E-value at a given score (linear interpolation of the table).
    pub fn evalue(&self, score: f64) -> f64 {
        interpolate(&self.cdf, score)
    }
}

/// Compute E-values for all hits using the final mask's E-value table.
pub fn annotate_hits(
    hits: &mut [Hit],
    pattern: &Pattern,
    mask: &ResolvedMask,
    bg: &Background,
    datavol: f64,
) {
    let table = EvalueTable::build(pattern, mask, bg, datavol);
    for hit in hits.iter_mut() {
        hit.evalue = Some(table.evalue(hit.score));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;
    use crate::pattern;
    use crate::region;
    use crate::search;

    #[test]
    fn test_evalue_computation() {
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

        let specs = region::parse_mask_specs("6,8 / !2,3 / *").unwrap();
        let mut masks = search::resolve_masks(&specs, &pat);
        let cutoffs = vec!["100%".to_string(), "100%".to_string(), "90%".to_string()];
        search::compute_thresholds(&ts, &pat, &mut masks, &cutoffs);

        // 1936 nt × 2 strands matches the C reference output.
        let datavol = 1936.0 * 2.0;
        let table = EvalueTable::build(&pat, masks.last().unwrap(), &bg, datavol);

        let ev_at_cutoff = table.evalue(59.7);
        eprintln!("E-value at score 59.7: {:.2e}", ev_at_cutoff);

        let ev_at_88 = table.evalue(88.0);
        eprintln!("E-value at score 88.0: {:.2e}", ev_at_88);

        assert!(
            table.evalue(80.0) < table.evalue(60.0),
            "E-value should decrease with higher scores"
        );

        assert!(
            ev_at_cutoff < 1.0,
            "E-value at cutoff should be < 1, got {}",
            ev_at_cutoff
        );
        assert!(ev_at_cutoff > 0.0, "E-value should be positive");
    }
}
