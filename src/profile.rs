use crate::types::TrainingSet;
use std::collections::BTreeMap;

/// Default log-zero value (lower bound for log of zero-frequency).
pub const LOG_ZERO: f64 = -20.0;

/// Number of single-nucleotide codes (A, T, G, C, gap, unknown).
pub const NT_CODES: usize = 6;
/// Number of dinucleotide codes for helix base-pair scoring.
/// 4 × 4 = 16 standard pairs, plus gap/unknown combinations → 24 in ERPIN.
pub const DINUC_CODES: usize = 24;

/// Nucleotide background frequencies.
#[derive(Debug, Clone)]
pub struct Background {
    /// Frequencies for A, T, G, C.
    pub freq: [f64; 4],
}

impl Default for Background {
    fn default() -> Self {
        Self {
            freq: [0.25, 0.25, 0.25, 0.25],
        }
    }
}

impl Background {
    /// Compute background frequencies from a sequence.
    pub fn from_sequence(data: &[u8]) -> Self {
        let mut counts = [0u64; 4];
        let mut total = 0u64;
        for &b in data {
            let idx = match b {
                b'A' => Some(0),
                b'T' => Some(1),
                b'G' => Some(2),
                b'C' => Some(3),
                _ => None,
            };
            if let Some(i) = idx {
                counts[i] += 1;
                total += 1;
            }
        }
        if total == 0 {
            return Self::default();
        }
        let freq = [
            counts[0] as f64 / total as f64,
            counts[1] as f64 / total as f64,
            counts[2] as f64 / total as f64,
            counts[3] as f64 / total as f64,
        ];
        Self { freq }
    }
}

/// Precomputed lookup table: ASCII byte → strand code.
/// A=0, T=1, G=2, C=3, gap=4, other=5.
static NT_STRAND_LUT: [u8; 256] = {
    let mut lut = [5u8; 256];
    lut[b'A' as usize] = 0;
    lut[b'a' as usize] = 0;
    lut[b'T' as usize] = 1;
    lut[b't' as usize] = 1;
    lut[b'G' as usize] = 2;
    lut[b'g' as usize] = 2;
    lut[b'C' as usize] = 3;
    lut[b'c' as usize] = 3;
    lut[b'-' as usize] = 4;
    lut
};

/// Encode a single nucleotide for strand profile indexing.
/// A=0, T=1, G=2, C=3, gap=4, other=5.
#[inline(always)]
pub fn nt_strand_code(b: u8) -> usize {
    NT_STRAND_LUT[b as usize] as usize
}

/// Precomputed 2D lookup table: (b5, b3) → helix dinucleotide code (0..23).
/// Indexed by strand codes (0..5), not ASCII bytes.
static DINUC_LUT: [[u8; 6]; 6] = {
    let mut lut = [[0u8; 6]; 6];
    let mut c5 = 0;
    while c5 < 6 {
        let mut c3 = 0;
        while c3 < 6 {
            lut[c5][c3] = if c5 < 4 && c3 < 4 {
                (c5 * 4 + c3) as u8
            } else {
                let a = if c5 >= 4 { c5 - 4 } else { 0 };
                let b = if c3 < 4 { c3 } else { if c3 < 6 { c3 - 4 } else { 0 } };
                (16 + a * 4 + b) as u8
            };
            c3 += 1;
        }
        c5 += 1;
    }
    lut
};

/// Encode a dinucleotide (base pair) for helix profile indexing.
/// Returns a code 0..23 representing the pair (b5, b3) where b5 is the 5'
/// base and b3 is the corresponding 3' base.
///
/// Standard pairs (0-15): 4 × nt_code(b5) + nt_code(b3)
/// Gap/unknown pairs get codes 16-23.
#[inline(always)]
pub fn nt_helix_code(b5: u8, b3: u8) -> usize {
    let c5 = NT_STRAND_LUT[b5 as usize] as usize;
    let c3 = NT_STRAND_LUT[b3 as usize] as usize;
    DINUC_LUT[c5][c3] as usize
}

/// Build a log-odds scoring profile for a strand region of the training set.
///
/// Returns a profile matrix of shape `[NT_CODES][region_len]`.
/// Each entry is `log2(observed_freq / background_freq)` with pseudo-counts.
pub fn build_strand_profile(
    training_set: &TrainingSet,
    columns: &[usize],
    background: &Background,
    pseudo_count_weight: f64,
    log_zero: f64,
) -> Vec<Vec<f64>> {
    let region_len = columns.len();
    let nseq = training_set.nseq as f64;

    // Count nucleotide frequencies at each position.
    let mut profile = vec![vec![0.0f64; region_len]; NT_CODES];

    for (pos, &col) in columns.iter().enumerate() {
        let mut counts = [0.0f64; NT_CODES];
        for seq in &training_set.sequences {
            let code = nt_strand_code(seq[col]);
            counts[code] += 1.0;
        }

        // Apply pseudo-counts and compute log-odds.
        for code in 0..4 {
            let observed = counts[code] / nseq;
            let smoothed = (1.0 - pseudo_count_weight) * observed
                + pseudo_count_weight * background.freq[code];
            let bg = background.freq[code];

            profile[code][pos] = if smoothed > 0.0 && bg > 0.0 {
                (smoothed / bg).ln() / std::f64::consts::LN_2
            } else {
                log_zero
            };
        }

        // Gap score.
        let gap_freq = counts[4] / nseq;
        profile[4][pos] = if gap_freq > 0.0 {
            gap_freq.ln() / std::f64::consts::LN_2
        } else {
            log_zero
        };

        // Unknown base score.
        profile[5][pos] = 0.0;
    }

    profile
}

/// Build a log-odds scoring profile for a helix region.
///
/// Takes the 5' and 3' column sets (which must have equal length).
/// Returns a profile matrix of shape `[DINUC_CODES][helix_len]`.
pub fn build_helix_profile(
    training_set: &TrainingSet,
    columns_5p: &[usize],
    columns_3p: &[usize],
    background: &Background,
    pseudo_count_weight: f64,
    log_zero: f64,
) -> Vec<Vec<f64>> {
    assert_eq!(
        columns_5p.len(),
        columns_3p.len(),
        "helix 5' and 3' must have equal length"
    );

    let helix_len = columns_5p.len();
    let nseq = training_set.nseq as f64;

    let mut profile = vec![vec![0.0f64; helix_len]; DINUC_CODES];

    for pos in 0..helix_len {
        let col5 = columns_5p[pos];
        // 3' strand is read in reverse order (antiparallel).
        let col3 = columns_3p[helix_len - 1 - pos];

        let mut counts = [0.0f64; DINUC_CODES];
        for seq in &training_set.sequences {
            let code = nt_helix_code(seq[col5], seq[col3]);
            counts[code] += 1.0;
        }

        // Log-odds for the 16 standard base pairs.
        for c5 in 0..4usize {
            for c3 in 0..4usize {
                let pair_code = c5 * 4 + c3;
                let observed = counts[pair_code] / nseq;
                let bg = background.freq[c5] * background.freq[c3];
                let smoothed =
                    (1.0 - pseudo_count_weight) * observed + pseudo_count_weight * bg;

                profile[pair_code][pos] = if smoothed > 0.0 && bg > 0.0 {
                    (smoothed / bg).ln() / std::f64::consts::LN_2
                } else {
                    log_zero
                };
            }
        }

        // Gap/unknown pair scores.
        for code in 16..DINUC_CODES {
            let freq = counts[code] / nseq;
            profile[code][pos] = if freq > 0.0 {
                freq.ln() / std::f64::consts::LN_2
            } else {
                log_zero
            };
        }
    }

    profile
}

/// Extract the column indices for a given element code from the training set
/// model. Returns the columns where `model[col] == code`, as a contiguous run.
pub fn columns_for_code(model: &[i32], code: i32) -> Vec<Vec<usize>> {
    let mut runs: Vec<Vec<usize>> = Vec::new();
    let mut current_run: Vec<usize> = Vec::new();

    for (col, &m) in model.iter().enumerate() {
        if m == code {
            current_run.push(col);
        } else if !current_run.is_empty() {
            runs.push(std::mem::take(&mut current_run));
        }
    }
    if !current_run.is_empty() {
        runs.push(current_run);
    }

    runs
}

/// Build profiles for all structural elements in a training set.
///
/// Returns maps from element code to their profile matrices.
pub fn build_all_profiles(
    training_set: &TrainingSet,
    background: &Background,
    pseudo_count_weight: f64,
    log_zero: f64,
) -> (BTreeMap<i32, Vec<Vec<f64>>>, BTreeMap<i32, Vec<Vec<f64>>>) {
    let mut strand_profiles = BTreeMap::new();
    let mut helix_profiles = BTreeMap::new();

    // Build strand profiles.
    for &code in &training_set.strand_codes {
        let runs = columns_for_code(&training_set.model, code);
        if let Some(cols) = runs.first() {
            let profile =
                build_strand_profile(training_set, cols, background, pseudo_count_weight, log_zero);
            strand_profiles.insert(code, profile);
        }
    }

    // Build helix profiles.
    for &code in &training_set.helix_codes {
        let runs = columns_for_code(&training_set.model, code);
        if runs.len() == 2 {
            let profile = build_helix_profile(
                training_set,
                &runs[0],
                &runs[1],
                background,
                pseudo_count_weight,
                log_zero,
            );
            helix_profiles.insert(code, profile);
        }
    }

    (strand_profiles, helix_profiles)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nt_strand_code() {
        assert_eq!(nt_strand_code(b'A'), 0);
        assert_eq!(nt_strand_code(b'T'), 1);
        assert_eq!(nt_strand_code(b'G'), 2);
        assert_eq!(nt_strand_code(b'C'), 3);
        assert_eq!(nt_strand_code(b'-'), 4);
        assert_eq!(nt_strand_code(b'N'), 5);
    }

    #[test]
    fn test_nt_helix_code() {
        // AA = 0*4+0 = 0, AT = 0*4+1 = 1, ..., CC = 3*4+3 = 15
        assert_eq!(nt_helix_code(b'A', b'A'), 0);
        assert_eq!(nt_helix_code(b'A', b'T'), 1);
        assert_eq!(nt_helix_code(b'C', b'C'), 15);
        // Gap pairs get codes >= 16
        assert!(nt_helix_code(b'-', b'A') >= 16);
    }

    #[test]
    fn test_columns_for_code() {
        let model = vec![1, 1, 1, 2, 2, 1, 1, 3, 3, 3];
        let runs = columns_for_code(&model, 1);
        assert_eq!(runs.len(), 2); // Two separate runs of code 1
        assert_eq!(runs[0], vec![0, 1, 2]);
        assert_eq!(runs[1], vec![5, 6]);
    }

    #[test]
    fn test_background_from_sequence() {
        let seq = b"AATTGGCC";
        let bg = Background::from_sequence(seq);
        assert!((bg.freq[0] - 0.25).abs() < 1e-10); // A
        assert!((bg.freq[1] - 0.25).abs() < 1e-10); // T
        assert!((bg.freq[2] - 0.25).abs() < 1e-10); // G
        assert!((bg.freq[3] - 0.25).abs() < 1e-10); // C
    }

    #[test]
    fn test_build_profiles_on_real_data() {
        let ts = crate::epn::parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
        let bg = Background::default();
        let (strand_profiles, helix_profiles) = build_all_profiles(&ts, &bg, 0.0002, LOG_ZERO);

        // Should have profiles for all strands and helices.
        assert_eq!(strand_profiles.len(), ts.nstrand);
        assert_eq!(helix_profiles.len(), ts.nhelix);

        // Each strand profile should have NT_CODES rows.
        for (_code, profile) in &strand_profiles {
            assert_eq!(profile.len(), NT_CODES);
        }

        // Each helix profile should have DINUC_CODES rows.
        for (_code, profile) in &helix_profiles {
            assert_eq!(profile.len(), DINUC_CODES);
        }
    }
}
