use crate::types::TrainingSet;
use std::collections::BTreeMap;

/// Default log-zero value (lower bound for log of zero-frequency).
pub const LOG_ZERO: f64 = -20.0;

/// Number of single-nucleotide codes (A, T, G, C, gap, unknown).
pub const NT_CODES: usize = 6;
/// Number of dinucleotide codes for helix base-pair scoring.
///
/// Matches C ERPIN's `NtHlxCode` layout (libsrc/ntcode.c::SetNtHlxCode):
///   - 0..15:  16 standard pairs `4*c5 + c3` for c5,c3 ∈ {A,T,G,C}
///   - 16..19: (N, A), (N, T), (N, G), (N, C)
///   - 20..23: (A, N), (T, N), (G, N), (C, N)
///   - 24:     "E"-marked positions (unrecognised / sequence edge) — scored at LOG_ZERO
///   - 25:     gap/(N, N)/everything else — scored at 0
pub const DINUC_CODES: usize = 26;
/// Alphabet length for bases (A, T, G, C).
const ALPHA_LEN: usize = 4;
/// Square of alphabet length, used for dinucleotide indexing.
const SQR_ALPHA_LEN: usize = ALPHA_LEN * ALPHA_LEN;

/// Default 4×4 strand substitution matrix from `libsrc/defaultSUM.c::DefaultStSum`.
/// Used by `SumImg` to mix observed counts with substitution-derived
/// pseudo-counts (matches C ERPIN's `-globstat` behaviour out of the box).
#[rustfmt::skip]
pub static DEFAULT_ST_SUM: [[f64; ALPHA_LEN]; ALPHA_LEN] = [
    [8.462e-01, 8.922e-02, 1.106e-01, 1.052e-01],
    [5.162e-02, 7.271e-01, 8.399e-02, 1.595e-01],
    [6.431e-02, 8.440e-02, 7.703e-01, 5.658e-02],
    [3.789e-02, 9.930e-02, 3.505e-02, 6.787e-01],
];

/// Default 16×16 helix substitution matrix from `libsrc/defaultSUM.c::DefaultHlxSum`.
/// Indexed by `4*c5 + c3` for both rows and columns.
#[rustfmt::skip]
pub static DEFAULT_HLX_SUM: [[f64; SQR_ALPHA_LEN]; SQR_ALPHA_LEN] = [
    [2.931e-01, 2.422e-03, 1.738e-02, 1.972e-02, 9.730e-03, 2.382e-03, 5.657e-03, 1.776e-02, 3.753e-02, 5.774e-03, 7.885e-02, 1.301e-03, 7.399e-03, 2.987e-03, 1.482e-03, 2.050e-03],
    [3.568e-02, 4.189e-01, 2.320e-02, 8.325e-02, 7.701e-02, 6.204e-02, 4.138e-02, 8.604e-02, 1.827e-02, 7.616e-02, 5.463e-02, 8.943e-02, 4.380e-02, 1.127e-01, 5.806e-02, 1.440e-01],
    [4.481e-02, 4.060e-03, 6.075e-01, 5.932e-02, 7.849e-03, 2.203e-03, 1.449e-02, 6.409e-02, 2.527e-02, 7.186e-03, 4.551e-02, 3.697e-03, 5.012e-03, 7.506e-03, 6.189e-03, 1.121e-02],
    [1.846e-02, 5.288e-03, 2.154e-02, 4.568e-01, 1.274e-03, 3.034e-03, 1.304e-03, 2.066e-02, 2.173e-03, 8.880e-03, 1.367e-02, 3.624e-03, 2.710e-03, 7.152e-03, 1.148e-03, 3.186e-02],
    [1.900e-01, 1.021e-01, 5.945e-02, 2.657e-02, 5.289e-01, 9.558e-02, 1.101e-01, 1.288e-01, 4.221e-02, 5.350e-02, 8.040e-02, 4.884e-02, 1.778e-01, 9.441e-02, 1.042e-01, 4.049e-02],
    [4.741e-03, 8.383e-03, 1.701e-03, 6.454e-03, 9.746e-03, 4.021e-01, 1.203e-02, 3.579e-02, 2.552e-03, 1.050e-02, 2.177e-02, 5.325e-03, 7.875e-03, 4.024e-02, 8.498e-03, 2.292e-01],
    [4.718e-02, 2.343e-02, 4.688e-02, 1.162e-02, 4.701e-02, 5.038e-02, 5.340e-01, 8.325e-02, 2.066e-02, 2.057e-02, 5.610e-02, 1.919e-02, 8.544e-02, 9.931e-02, 3.488e-02, 1.412e-02],
    [6.505e-03, 2.139e-03, 9.105e-03, 8.085e-03, 2.415e-03, 6.584e-03, 3.655e-03, 7.409e-02, 2.006e-03, 4.048e-03, 2.133e-03, 1.649e-03, 4.516e-03, 2.848e-02, 1.367e-03, 7.037e-03],
    [1.154e-01, 3.813e-03, 3.013e-02, 7.137e-03, 6.646e-03, 3.940e-03, 7.617e-03, 1.684e-02, 6.502e-01, 1.126e-02, 3.890e-02, 9.557e-03, 1.518e-02, 1.306e-02, 2.039e-03, 3.502e-03],
    [5.581e-02, 4.996e-02, 2.694e-02, 9.171e-02, 2.648e-02, 5.099e-02, 2.384e-02, 1.068e-01, 3.541e-02, 5.781e-01, 4.201e-02, 3.942e-02, 1.271e-02, 4.210e-02, 1.648e-02, 1.999e-02],
    [8.158e-02, 3.836e-03, 1.827e-02, 1.511e-02, 4.260e-03, 1.131e-02, 6.959e-03, 6.027e-03, 1.309e-02, 4.497e-03, 2.515e-01, 3.129e-03, 3.427e-03, 4.063e-03, 6.148e-03, 4.651e-03],
    [5.132e-02, 2.395e-01, 5.659e-02, 1.528e-01, 9.872e-02, 1.056e-01, 9.078e-02, 1.777e-01, 1.227e-01, 1.609e-01, 1.193e-01, 6.730e-01, 5.389e-02, 2.428e-01, 1.192e-01, 1.256e-01],
    [5.843e-03, 2.348e-03, 1.535e-03, 2.286e-03, 7.192e-03, 3.124e-03, 8.090e-03, 9.739e-03, 3.899e-03, 1.038e-03, 2.616e-03, 1.079e-03, 2.662e-01, 2.448e-02, 7.332e-03, 3.354e-03],
    [1.734e-03, 4.441e-03, 1.690e-03, 4.436e-03, 2.807e-03, 1.174e-02, 6.913e-03, 4.516e-02, 2.466e-03, 2.529e-03, 2.280e-03, 3.571e-03, 1.800e-02, 1.363e-01, 2.421e-03, 1.536e-02],
    [4.693e-02, 1.248e-01, 7.603e-02, 3.884e-02, 1.690e-01, 1.352e-01, 1.325e-01, 1.183e-01, 2.101e-02, 5.401e-02, 1.882e-01, 9.567e-02, 2.940e-01, 1.321e-01, 6.289e-01, 1.094e-01],
    [9.584e-04, 4.569e-03, 2.033e-03, 1.591e-02, 9.694e-04, 5.383e-02, 7.917e-04, 8.983e-03, 5.326e-04, 9.670e-04, 2.101e-03, 1.488e-03, 1.986e-03, 1.237e-02, 1.615e-03, 2.381e-01],
];

/// Apply a substitution matrix to a per-column count profile in place.
///
/// Mirrors `libsrc/sum.c::SumImg`. For each column j and each row i:
/// `tmp[i] = sum_k Sbstm[i][k] * prof[k][j]`, then
/// `prof[i][j] = (1 - weight) * prof[i][j] + weight * tmp[i]`.
fn sum_img(sbstm: &[&[f64]], dim: usize, prof: &mut [Vec<f64>], len: usize, weight: f64) {
    debug_assert!((0.0..=1.0).contains(&weight));
    let mut tmp = vec![0.0f64; dim];
    for j in 0..len {
        for v in tmp.iter_mut() {
            *v = 0.0;
        }
        for i in 0..dim {
            let row = sbstm[i];
            let mut acc = 0.0f64;
            for k in 0..dim {
                acc += row[k] * prof[k][j];
            }
            tmp[i] = acc;
        }
        for i in 0..dim {
            prof[i][j] = (1.0 - weight) * prof[i][j] + weight * tmp[i];
        }
    }
}

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

/// ASCII byte → helix-class code:
///   0=A, 1=T, 2=G, 3=C, 4=N, 5=E (sequence-edge marker), 6=anything else.
/// Lowercase bases and U/u (RNA) fold into the same classes as their
/// uppercase DNA equivalents, matching C ERPIN's behaviour after FASTA load.
pub static NT_HELIX_CLASS_LUT: [u8; 256] = {
    let mut lut = [6u8; 256];
    lut[b'A' as usize] = 0;
    lut[b'a' as usize] = 0;
    lut[b'T' as usize] = 1;
    lut[b't' as usize] = 1;
    lut[b'U' as usize] = 1;
    lut[b'u' as usize] = 1;
    lut[b'G' as usize] = 2;
    lut[b'g' as usize] = 2;
    lut[b'C' as usize] = 3;
    lut[b'c' as usize] = 3;
    lut[b'N' as usize] = 4;
    lut[b'n' as usize] = 4;
    lut[b'E' as usize] = 5;
    lut[b'e' as usize] = 5;
    lut
};

/// 7×7 → helix dinucleotide-row LUT, indexed by helix-class codes from
/// [`NT_HELIX_CLASS_LUT`]. Mirrors `libsrc/ntcode.c::SetNtHlxCode`:
///   - (ATGC, ATGC) → 4*c5 + c3 ∈ 0..15
///   - (N, ATGC) → 16 + c3
///   - (ATGC, N) → 20 + c5
///   - any (E, _) or (_, E) → 24 (LOG_ZERO row)
///   - everything else (incl. (N, N) and gaps) → 25 (zero row)
pub static DINUC_LUT: [[u8; 7]; 7] = {
    let mut lut = [[25u8; 7]; 7]; // default: zero row (matches C's FillsMat 25)
    // Standard ATGC × ATGC pairs.
    let mut c5 = 0;
    while c5 < 4 {
        let mut c3 = 0;
        while c3 < 4 {
            lut[c5][c3] = (c5 * 4 + c3) as u8;
            c3 += 1;
        }
        c5 += 1;
    }
    // (N, ATGC) → 16 + c3.
    let mut c3 = 0;
    while c3 < 4 {
        lut[4][c3] = (16 + c3) as u8;
        c3 += 1;
    }
    // (ATGC, N) → 20 + c5.
    let mut c5 = 0;
    while c5 < 4 {
        lut[c5][4] = (20 + c5) as u8;
        c5 += 1;
    }
    // (E, *) and (*, E) → 24 (LOG_ZERO row).
    let mut k = 0;
    while k < 7 {
        lut[5][k] = 24;
        lut[k][5] = 24;
        k += 1;
    }
    lut
};

/// Encode a dinucleotide (base pair) for helix profile indexing.
/// Returns a code 0..25 — see `DINUC_CODES` for the full row layout.
#[inline(always)]
pub fn nt_helix_code(b5: u8, b3: u8) -> usize {
    let c5 = NT_HELIX_CLASS_LUT[b5 as usize] as usize;
    let c3 = NT_HELIX_CLASS_LUT[b3 as usize] as usize;
    DINUC_LUT[c5][c3] as usize
}

/// Build a log-odds scoring profile for a strand region of the training set.
///
/// Mirrors `libsrc/profsSM.c::GetWeightsSM`: count A/T/G/C with N spread
/// uniformly, gap as a separate row, then mix with substitution-matrix
/// pseudo-counts via [`sum_img`], normalize per column, and emit
/// `ln(P/sum) - log1mPg - ln(bg[i])` for the four bases (with the gap and
/// unknown rows on different baselines). Output rows: 0..3 = A,T,G,C;
/// 4 = X (gap, indexed by `nt_strand_code` as 4); 5 = N (unknown).
pub fn build_strand_profile(
    training_set: &TrainingSet,
    columns: &[usize],
    background: &Background,
    pseudo_count_weight: f64,
    log_zero: f64,
) -> Vec<Vec<f64>> {
    let region_len = columns.len();
    let nbstr = training_set.nseq;
    let eps = log_zero.exp();

    // Initial counts (height = ALPHA_LEN + 2 → 6 rows).
    let mut profile = vec![vec![0.0f64; region_len]; NT_CODES];
    let mut totalgaps = 0u64;

    for (pos, &col) in columns.iter().enumerate() {
        let (mut na, mut nt, mut ng, mut nc, mut nn, mut gaps) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
        for seq in &training_set.sequences {
            // Match C ERPIN's switch: only A/T/G/C/-/N are accepted; anything
            // else triggers ioError. We treat unknown bytes as N for safety.
            match seq[col] {
                b'A' | b'a' => na += 1,
                b'T' | b't' | b'U' | b'u' => nt += 1,
                b'G' | b'g' => ng += 1,
                b'C' | b'c' => nc += 1,
                b'-' => gaps += 1,
                _ => nn += 1, // includes N and any other char
            }
        }
        let u = nn as f64 / ALPHA_LEN as f64;
        let ncar = (nbstr as i64 - gaps) as f64;
        profile[0][pos] = na as f64 + u;
        profile[1][pos] = nt as f64 + u;
        profile[2][pos] = ng as f64 + u;
        profile[3][pos] = nc as f64 + u;
        profile[4][pos] = gaps as f64;
        profile[5][pos] = ncar / ALPHA_LEN as f64;
        totalgaps += gaps as u64;
    }

    // Apply substitution-matrix pseudo-counts to the first ALPHA_LEN rows.
    let st_sum_rows: [&[f64]; ALPHA_LEN] = [
        &DEFAULT_ST_SUM[0], &DEFAULT_ST_SUM[1], &DEFAULT_ST_SUM[2], &DEFAULT_ST_SUM[3],
    ];
    sum_img(&st_sum_rows, ALPHA_LEN, &mut profile, region_len, pseudo_count_weight);

    // Floor cells in the first ALPHA_LEN+2 rows (matches C's `i < height`,
    // which is ALPHA_LEN + 2 = 6 — our entire profile).
    for row in profile.iter_mut() {
        for v in row.iter_mut() {
            if *v < eps {
                *v = eps;
            }
        }
    }

    // Constants matching C's GetWeightsSM (Pg = 0.5).
    let mut log1m_pg = 0.5f64.ln();
    let mut log_pg = 0.5f64.ln();
    let log_po = (1.0 / ALPHA_LEN as f64).ln();
    let new_log_data_freqs = [
        background.freq[0].ln(),
        background.freq[1].ln(),
        background.freq[2].ln(),
        background.freq[3].ln(),
    ];

    // C: `if (totalgaps == 0) log1mPg = logPg = 0.0`.
    if totalgaps == 0 {
        log1m_pg = 0.0;
        log_pg = 0.0;
    }

    for j in 0..region_len {
        // sum is over rows 0..=ALPHA_LEN (i.e., A,T,G,C,X — indices 0..=4).
        let mut sum = 0.0f64;
        for i in 0..=ALPHA_LEN {
            sum += profile[i][j];
        }
        for i in 0..ALPHA_LEN {
            profile[i][j] = (profile[i][j] / sum).ln() - log1m_pg - new_log_data_freqs[i];
        }
        profile[4][j] = (profile[4][j] / sum).ln() - log_pg;
        profile[5][j] = (profile[5][j] / sum).ln() - log1m_pg - log_po;
    }

    profile
}

/// Build a log-odds scoring profile for a helix region.
///
/// Mirrors `libsrc/profsSM.c::GetCorrelsSM`: counts dinucleotide pairs from
/// the training set (with N spread uniformly across ATGC for whichever
/// base in the pair is N), applies substitution-matrix pseudo-counts via
/// [`sum_img`] over the 16 standard pairs, normalises per column, fills in
/// the 8 (N, X) and (X, N) rows as column averages of the 16 standard
/// pairs, and emits `ln(P) - (ln(bg[c5]) + ln(bg[c3]))` for the standard
/// pairs (and `ln(P) - ln(1/16)` for the N rows).
///
/// Layout matches `nt_helix_code` / `DINUC_LUT`: rows 0..15 are
/// `4*c5 + c3` for the 16 standard pairs, rows 16..19 are (X, A/T/G/C),
/// rows 20..23 are (A/T/G/C, X). Total 24 rows = `DINUC_CODES`.
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
    let eps = log_zero.exp();
    let alphalen_inv = 1.0 / ALPHA_LEN as f64;
    let sqr_alphalen_inv = 1.0 / SQR_ALPHA_LEN as f64;

    let mut profile = vec![vec![0.0f64; helix_len]; DINUC_CODES];

    for pos in 0..helix_len {
        let col5 = columns_5p[pos];
        // 3' strand is read in reverse order (antiparallel).
        let col3 = columns_3p[helix_len - 1 - pos];

        for seq in &training_set.sequences {
            let b5 = seq[col5];
            let b3 = seq[col3];
            // Decode each base to {A=0, T=1, G=2, C=3, N/other=N marker}.
            // Match C's switch: only ATGCN (and gap not handled in helix).
            let i1 = match b5 {
                b'A' | b'a' => Some(0usize),
                b'T' | b't' | b'U' | b'u' => Some(1),
                b'G' | b'g' => Some(2),
                b'C' | b'c' => Some(3),
                _ => None, // N, gap, or anything else
            };
            let i2 = match b3 {
                b'A' | b'a' => Some(0usize),
                b'T' | b't' | b'U' | b'u' => Some(1),
                b'G' | b'g' => Some(2),
                b'C' | b'c' => Some(3),
                _ => None,
            };
            match (i1, i2) {
                (Some(a), Some(b)) => {
                    profile[a * ALPHA_LEN + b][pos] += 1.0;
                }
                (None, Some(b)) => {
                    for a in 0..ALPHA_LEN {
                        profile[a * ALPHA_LEN + b][pos] += alphalen_inv;
                    }
                }
                (Some(a), None) => {
                    for b in 0..ALPHA_LEN {
                        profile[a * ALPHA_LEN + b][pos] += alphalen_inv;
                    }
                }
                (None, None) => {
                    for cell in 0..SQR_ALPHA_LEN {
                        profile[cell][pos] += sqr_alphalen_inv;
                    }
                }
            }
        }
    }

    // Apply substitution-matrix pseudo-counts to the 16 standard-pair rows.
    let mut hlx_rows: [&[f64]; SQR_ALPHA_LEN] = [&[]; SQR_ALPHA_LEN];
    for (i, row) in hlx_rows.iter_mut().enumerate() {
        *row = &DEFAULT_HLX_SUM[i];
    }
    sum_img(&hlx_rows, SQR_ALPHA_LEN, &mut profile, helix_len, pseudo_count_weight);

    // Floor the 16 standard-pair rows.
    for i in 0..SQR_ALPHA_LEN {
        for j in 0..helix_len {
            if profile[i][j] < eps {
                profile[i][j] = eps;
            }
        }
    }

    // Normalise each column over the 16 standard pairs.
    for j in 0..helix_len {
        let mut sum = 0.0f64;
        for i in 0..SQR_ALPHA_LEN {
            sum += profile[i][j];
        }
        for i in 0..SQR_ALPHA_LEN {
            profile[i][j] /= sum;
        }
    }

    // Rows 16..=19: (N, A|T|G|C). Average over the 4 c5 values for each c3.
    for j in 0..helix_len {
        for i in 0..ALPHA_LEN {
            let mut acc = 0.0f64;
            for k in 0..ALPHA_LEN {
                acc += profile[ALPHA_LEN * k + i][j];
            }
            profile[SQR_ALPHA_LEN + i][j] = acc * alphalen_inv;
        }
    }

    // Rows 20..=23: (A|T|G|C, N). Average over the 4 c3 values for each c5.
    for j in 0..helix_len {
        for i in 0..ALPHA_LEN {
            let mut acc = 0.0f64;
            for k in 0..ALPHA_LEN {
                acc += profile[ALPHA_LEN * i + k][j];
            }
            profile[SQR_ALPHA_LEN + ALPHA_LEN + i][j] = acc * alphalen_inv;
        }
    }

    // Convert the 24 "real" rows to log-odds against the background.
    let log_bg = [
        background.freq[0].ln(),
        background.freq[1].ln(),
        background.freq[2].ln(),
        background.freq[3].ln(),
    ];
    let log_po = sqr_alphalen_inv.ln(); // ln(1/16)
    for i in 0..SQR_ALPHA_LEN {
        let u = log_bg[i / ALPHA_LEN] + log_bg[i % ALPHA_LEN];
        for j in 0..helix_len {
            profile[i][j] = profile[i][j].ln() - u;
        }
    }
    for i in SQR_ALPHA_LEN..(SQR_ALPHA_LEN + 2 * ALPHA_LEN) {
        for j in 0..helix_len {
            profile[i][j] = profile[i][j].ln() - log_po;
        }
    }

    // Auxiliary rows that match `libsrc/profsSM.c::GetCorrelsSM` allocation:
    //   row 24 ("E"-marked): LOG_ZERO contribution per column.
    //   row 25 (default fall-through, incl. (N, N) and gap pairs): 0 contribution.
    for j in 0..helix_len {
        profile[24][j] = log_zero;
        profile[25][j] = 0.0;
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
        let ts = crate::epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
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
