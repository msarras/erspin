use crate::profile::Background;
use crate::types::*;

/// Discrete histogram with fixed-width bins.
#[derive(Debug, Clone)]
struct Histogram {
    /// Probability in each bin.
    vals: Vec<f64>,
    /// Left edge of bin 0.
    origin: f64,
    /// Bin width.
    bin_width: f64,
}

impl Histogram {
    fn new(origin: f64, bin_width: f64, nbins: usize) -> Self {
        Self {
            vals: vec![0.0; nbins],
            origin,
            bin_width,
        }
    }

    /// Bin index for a given score.
    fn bin_for(&self, score: f64) -> Option<usize> {
        let idx = ((score - self.origin) / self.bin_width).floor() as isize;
        if idx >= 0 && (idx as usize) < self.vals.len() {
            Some(idx as usize)
        } else {
            None
        }
    }

    fn nbins(&self) -> usize {
        self.vals.len()
    }

}

/// Convolve two histograms: result[k] = sum_j h1[j] * h2[k-j].
fn convolve(h1: &Histogram, h2: &Histogram) -> Histogram {
    let n1 = h1.nbins();
    let n2 = h2.nbins();
    let n_out = n1 + n2 - 1;
    let mut out = Histogram::new(h1.origin + h2.origin, h1.bin_width, n_out);

    for i in 0..n1 {
        if h1.vals[i] == 0.0 {
            continue;
        }
        for j in 0..n2 {
            out.vals[i + j] += h1.vals[i] * h2.vals[j];
        }
    }

    out
}

/// Compute the right-tail CDF from a histogram.
/// cdf[i] = P(score >= score_at(i)) = sum(vals[j] for j >= i).
fn right_tail_cdf(hist: &Histogram) -> Histogram {
    let mut cdf = hist.clone();
    let n = cdf.nbins();
    if n == 0 {
        return cdf;
    }
    // Accumulate from right to left.
    for i in (0..n - 1).rev() {
        cdf.vals[i] += cdf.vals[i + 1];
    }
    cdf
}

/// Interpolate a value from a histogram (treated as CDF).
fn interpolate(cdf: &Histogram, score: f64) -> f64 {
    if cdf.nbins() == 0 {
        return 0.0;
    }
    let x = (score - cdf.origin) / cdf.bin_width - 0.25;
    if x <= 0.0 {
        return cdf.vals[0];
    }
    let idx = x.floor() as usize;
    if idx >= cdf.nbins() - 1 {
        return *cdf.vals.last().unwrap_or(&0.0);
    }
    let frac = x - idx as f64;
    cdf.vals[idx] * (1.0 - frac) + cdf.vals[idx + 1] * frac
}

/// Extreme value probability: P(max of N trials > x) given P(single > x) = p.
/// = 1 - (1-p)^N
fn evprob(p: f64, n: usize) -> f64 {
    if p <= 0.0 || n == 0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let pn = p * n as f64;
    if pn < 1e-6 {
        // Small p*N: use expansion to avoid floating point issues.
        // 1 - (1-p)^N ≈ N*p - N*(N-1)*p^2/2 + ...
        let nf = n as f64;
        let term1 = nf * p;
        let term2 = nf * (nf - 1.0) * p * p / 2.0;
        let term3 = nf * (nf - 1.0) * (nf - 2.0) * p * p * p / 6.0;
        (term1 - term2 + term3).max(0.0)
    } else {
        1.0 - (1.0 - p).powi(n as i32)
    }
}

const BIN_WIDTH: f64 = 0.5;
const SCORE_MIN: f64 = -40.0;
const SCORE_MAX: f64 = 40.0;

/// Build a score histogram for a single strand column given background frequencies.
fn strand_column_histogram(profile: &[Vec<f64>], col: usize, bg: &Background) -> Histogram {
    let nbins = ((SCORE_MAX - SCORE_MIN) / BIN_WIDTH).ceil() as usize;
    let mut hist = Histogram::new(SCORE_MIN, BIN_WIDTH, nbins);

    // Standard nucleotides (A=0, T=1, G=2, C=3).
    for nt in 0..4 {
        let score = profile[nt][col];
        let weight = bg.freq[nt];
        if let Some(bin) = hist.bin_for(score) {
            hist.vals[bin] += weight;
        }
    }

    hist
}

/// Build the score histogram for a full strand element.
fn strand_histogram(strand: &Strand, bg: &Background) -> Histogram {
    // Start with the first column's histogram.
    let mut hist = strand_column_histogram(&strand.profile, 0, bg);

    // Convolve with each subsequent column.
    for col in 1..strand.max_len {
        let col_hist = strand_column_histogram(&strand.profile, col, bg);
        hist = convolve(&hist, &col_hist);
    }

    // Normalize.
    let total: f64 = hist.vals.iter().sum();
    if total > 0.0 {
        for v in &mut hist.vals {
            *v /= total;
        }
    }

    hist
}

/// Build a score histogram for a single helix column given background frequencies.
fn helix_column_histogram(profile: &[Vec<f64>], col: usize, bg: &Background) -> Histogram {
    let nbins = ((SCORE_MAX - SCORE_MIN) / BIN_WIDTH).ceil() as usize;
    let mut hist = Histogram::new(SCORE_MIN, BIN_WIDTH, nbins);

    // Dinucleotide pairs: 4 × 4 = 16 standard pairs.
    for b5 in 0..4usize {
        for b3 in 0..4usize {
            let code = b5 * 4 + b3;
            let score = profile[code][col];
            let weight = bg.freq[b5] * bg.freq[b3];
            if let Some(bin) = hist.bin_for(score) {
                hist.vals[bin] += weight;
            }
        }
    }

    hist
}

/// Build the score histogram for a full helix element.
fn helix_histogram(helix: &Helix, bg: &Background) -> Histogram {
    let mut hist = helix_column_histogram(&helix.profile, 0, bg);

    for col in 1..helix.helix_len {
        let col_hist = helix_column_histogram(&helix.profile, col, bg);
        hist = convolve(&hist, &col_hist);
    }

    let total: f64 = hist.vals.iter().sum();
    if total > 0.0 {
        for v in &mut hist.vals {
            *v /= total;
        }
    }

    hist
}

/// Build the combined score histogram for a mask (all elements convolved).
fn mask_histogram(pattern: &Pattern, mask: &ResolvedMask, bg: &Background) -> Histogram {
    let mut histograms: Vec<Histogram> = Vec::new();

    for &si in &mask.st_indices {
        histograms.push(strand_histogram(&pattern.strands[si], bg));
    }
    for &hi in &mask.hx_indices {
        histograms.push(helix_histogram(&pattern.helices[hi], bg));
    }

    if histograms.is_empty() {
        return Histogram::new(0.0, BIN_WIDTH, 1);
    }

    // Convolve all element histograms together.
    let mut combined = histograms.remove(0);
    for h in &histograms {
        combined = convolve(&combined, h);
    }

    // Normalize.
    let total: f64 = combined.vals.iter().sum();
    if total > 0.0 {
        for v in &mut combined.vals {
            *v /= total;
        }
    }

    combined
}

/// Precomputed E-value lookup table for a mask.
pub struct EvalueTable {
    /// Right-tail CDF scaled by datavol and extreme value correction.
    cdf: Histogram,
}

impl EvalueTable {
    /// Build E-value table for the given mask.
    ///
    /// `datavol` is the total number of nucleotide positions to scan
    /// (sum of sequence lengths). For double-strand search, multiply by 2.
    pub fn build(
        pattern: &Pattern,
        mask: &ResolvedMask,
        bg: &Background,
        datavol: f64,
    ) -> Self {
        let hist = mask_histogram(pattern, mask, bg);
        let mut cdf = right_tail_cdf(&hist);

        // Scale CDF by database volume and extreme value correction.
        let ncfg = mask.configs.len();
        for i in 0..cdf.nbins() {
            let p = cdf.vals[i]; // P(random score >= this bin's score)
            let ev_p = evprob(p, ncfg);
            cdf.vals[i] = datavol * ev_p;
        }

        Self { cdf }
    }

    /// Look up the E-value for a given score.
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

        // Build E-value table for the final mask.
        // Use 1936 nt * 2 (both strands) as the C code reports.
        let datavol = 1936.0 * 2.0;
        let table = EvalueTable::build(&pat, masks.last().unwrap(), &bg, datavol);

        // The C code reports E-value at cutoff 59.7 = 4.71e-16.
        let ev_at_cutoff = table.evalue(59.7);
        eprintln!("E-value at score 59.7: {:.2e}", ev_at_cutoff);

        // E-value should be very small for high scores.
        let ev_at_88 = table.evalue(88.0);
        eprintln!("E-value at score 88.0: {:.2e}", ev_at_88);

        // Sanity: E-value should decrease with increasing score.
        assert!(
            table.evalue(80.0) < table.evalue(60.0),
            "E-value should decrease with higher scores"
        );

        // E-value at the cutoff score should be small but non-zero.
        assert!(
            ev_at_cutoff < 1.0,
            "E-value at cutoff should be < 1, got {}",
            ev_at_cutoff
        );
        assert!(ev_at_cutoff > 0.0, "E-value should be positive");
    }
}
