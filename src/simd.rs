//! SIMD-accelerated scoring routines.
//!
//! These functions compute the same results as the scalar versions in scoring.rs
//! but use loop reordering and SIMD intrinsics for throughput.
//!
//! Strategy: process multiple scan positions simultaneously by reordering loops
//! from position-major (for s { for j }) to column-major (for j { for s }).

use crate::profile::{nt_helix_code, nt_strand_code, LOG_ZERO};
use crate::types::*;

/// Precomputed lookup table: low nibble of ASCII byte → nucleotide code (0-5).
///
/// Works for both uppercase and lowercase since A/a share nibble 1, T/t share 4, etc.
/// Index:  0  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
/// Value:  5  0  5  3  1  5  5  2  5  5  5  5  5  4  5  5
const NT_NIBBLE_LUT: [u8; 16] = [5, 0, 5, 3, 1, 5, 5, 2, 5, 5, 5, 5, 5, 4, 5, 5];

/// Compute score table for a no-gap strand using column-major loop order.
///
/// This reorders the computation so the inner loop scans across positions
/// for a fixed profile column, which is much more SIMD-friendly.
pub fn compute_strand_score_table_nogap(seq: &[u8], strand: &Strand) -> Vec<f32> {
    let max_len = strand.max_len;
    let scan_len = if seq.len() >= max_len {
        seq.len() - max_len + 1
    } else {
        return Vec::new();
    };

    // Precompute per-column lookup: for each column j, store the 6 profile values
    // as f32 in a small array indexed by nucleotide code.
    let mut col_vals = vec![[0.0f32; 8]; max_len]; // padded to 8 for alignment
    for j in 0..max_len {
        for code in 0..6 {
            col_vals[j][code] = strand.profile[code][j] as f32;
        }
    }

    let mut scores = vec![0.0f32; scan_len];

    // Column-major: for each profile column j, accumulate across all positions.
    for j in 0..max_len {
        let vals = &col_vals[j];

        // Precompute codes for this column offset across all positions.
        // seq[s + j] for s in 0..scan_len → codes stored as bytes.
        let seq_slice = &seq[j..j + scan_len];

        // Process 8 positions at a time using scalar code that the compiler
        // can auto-vectorize.
        let chunks = scan_len / 8;
        let remainder = scan_len % 8;

        for chunk in 0..chunks {
            let base = chunk * 8;
            // Unroll 8 iterations to help auto-vectorization.
            for k in 0..8 {
                let code = NT_NIBBLE_LUT[(seq_slice[base + k] & 0x0F) as usize];
                scores[base + k] += vals[code as usize];
            }
        }

        // Handle remainder.
        let base = chunks * 8;
        for k in 0..remainder {
            let code = NT_NIBBLE_LUT[(seq_slice[base + k] & 0x0F) as usize];
            scores[base + k] += vals[code as usize];
        }
    }

    scores
}

/// Compute score table for a helix using column-major loop order.
///
/// For each gap variant k and profile column j, accumulates scores across all positions.
pub fn compute_helix_score_table(seq: &[u8], helix: &Helix) -> Vec<Vec<f32>> {
    let scan_len = if seq.len() >= helix.max_len {
        seq.len() - helix.max_len + 1
    } else {
        return vec![vec![]; helix.max_gaps + 1];
    };

    let num_gaps = helix.max_gaps + 1;
    let hlen = helix.helix_len;

    // Precompute per-column lookup: 24 profile values as f32.
    let mut col_vals = vec![[0.0f32; 24]; hlen];
    for j in 0..hlen {
        for code in 0..24 {
            col_vals[j][code] = helix.profile[code][j] as f32;
        }
    }

    let mut table = vec![vec![0.0f32; scan_len]; num_gaps];

    for k in 0..num_gaps {
        let scores = &mut table[k];
        // Zero the scores for this gap variant.
        for v in scores.iter_mut() {
            *v = 0.0;
        }

        // Column-major: for each profile column j, accumulate across positions.
        for j in 0..hlen {
            let vals = &col_vals[j];
            // For position s:
            //   left_base = seq[s + j]
            //   right_base = seq[s + helix.min_len - 1 + k - j]
            // The right offset from s is: (helix.min_len - 1 + k - j)
            let right_offset = helix.min_len - 1 + k - j;

            // Process positions in chunks for auto-vectorization.
            let chunks = scan_len / 8;
            let remainder = scan_len % 8;

            for chunk in 0..chunks {
                let base = chunk * 8;
                for i in 0..8 {
                    let s = base + i;
                    let left_base = seq[s + j];
                    let right_base = seq[s + right_offset];
                    let code = nt_helix_code(left_base, right_base);
                    scores[s] += vals[code];
                }
            }

            let base = chunks * 8;
            for i in 0..remainder {
                let s = base + i;
                let left_base = seq[s + j];
                let right_base = seq[s + right_offset];
                let code = nt_helix_code(left_base, right_base);
                scores[s] += vals[code];
            }
        }
    }

    table
}

/// Compute score table for a gapped strand using multi-position DP.
///
/// Runs W independent DP instances in parallel across scan positions.
/// Each "lane" processes a different scan position through the same DP structure.
pub fn compute_strand_score_table_gapped(seq: &[u8], strand: &Strand) -> Vec<Vec<f32>> {
    let max_len = strand.max_len;
    let max_gaps = strand.max_gaps;
    let min_len = strand.min_len;
    let scan_len = if seq.len() >= max_len {
        seq.len() - max_len + 1
    } else {
        return vec![vec![]; max_gaps + 1];
    };

    let num_gaps = max_gaps + 1;
    let mut table = vec![vec![0.0f32; scan_len]; num_gaps];

    let prof_len = max_len;
    let seq_len = max_len;
    let dp_cols = prof_len + 1;

    // Precompute flat f32 profile for the DP.
    let mut flat_profile = vec![0.0f32; 6 * prof_len];
    for code in 0..6 {
        for j in 0..prof_len {
            flat_profile[code * prof_len + j] = strand.profile[code][j] as f32;
        }
    }
    let gap_profile = &flat_profile[4 * prof_len..5 * prof_len];

    // Process W positions at a time.
    const W: usize = 4;

    let full_chunks = scan_len / W;
    let remainder = scan_len % W;

    // Pre-allocate interleaved DP buffer.
    // Layout: dp_buf[(i * dp_cols + j) * W + lane]
    let dp_size = (seq_len + 1) * dp_cols;
    let mut dp_buf = vec![f32::NEG_INFINITY; W * dp_size];
    let mut scores_buf = vec![0.0f32; W * num_gaps];

    for chunk in 0..full_chunks {
        let base_s = chunk * W;

        // Reset and fill W DP tables in parallel.
        multi_dp_w::<W>(
            seq, base_s, seq_len, prof_len, max_gaps, min_len,
            &flat_profile, gap_profile, dp_cols, dp_size,
            &mut dp_buf, &mut scores_buf,
        );

        // Write results.
        for lane in 0..W {
            for k in 0..num_gaps {
                table[k][base_s + lane] = scores_buf[lane * num_gaps + k];
            }
        }
    }

    // Handle remainder positions one at a time.
    if remainder > 0 {
        let dp_one_size = (seq_len + 1) * dp_cols;
        let mut align = vec![f64::NEG_INFINITY; dp_one_size];
        let mut sc = vec![0.0f64; num_gaps];

        let flat64: Vec<f64> = flat_profile.iter().map(|&v| v as f64).collect();
        let gap64: Vec<f64> = gap_profile.iter().map(|&v| v as f64).collect();

        for r in 0..remainder {
            let s = full_chunks * W + r;
            strand_dp_scalar(
                &seq[s..], seq_len, prof_len, max_gaps, min_len,
                &flat64, &gap64, &mut align, dp_cols, &mut sc,
            );
            for k in 0..num_gaps {
                table[k][s] = sc[k] as f32;
            }
        }
    }

    table
}

/// Run W independent DP instances on consecutive scan positions.
///
/// Uses interleaved SOA layout: dp_buf[(i * dp_cols + j) * W + lane]
/// so that all W lanes for the same cell are adjacent in memory,
/// enabling the compiler to auto-vectorize the inner lane loop.
fn multi_dp_w<const W: usize>(
    seq: &[u8],
    base_s: usize,
    seq_len: usize,
    prof_len: usize,
    max_gaps: usize,
    min_len: usize,
    flat_profile: &[f32],
    gap_profile: &[f32],
    dp_cols: usize,
    _dp_size: usize,
    dp_buf: &mut [f32],
    scores_buf: &mut [f32],
) {
    let num_gaps = max_gaps + 1;
    // Interleaved layout: cell(i, j) for lane L is at [(i * dp_cols + j) * W + L]
    let total_cells = (seq_len + 1) * dp_cols;

    // Reset band entries for all lanes (interleaved).
    // First, fill everything with NEG_INFINITY.
    for idx in 0..total_cells * W {
        dp_buf[idx] = f32::NEG_INFINITY;
    }

    // align[0][0] = 0 for all lanes.
    for lane in 0..W {
        dp_buf[0 * W + lane] = 0.0;
    }

    // First row: gaps in sequence (same for all lanes since profile is shared).
    for j in 1..=max_gaps.min(prof_len) {
        let prev_base = j - 1;
        let gap_val = gap_profile[prev_base];
        for lane in 0..W {
            dp_buf[j * W + lane] = dp_buf[prev_base * W + lane] + gap_val;
        }
    }

    // Diagonal: match — each lane has different sequence bases.
    for row in 1..=seq_len {
        let cell_idx = row * dp_cols + row;
        let prev_cell_idx = (row - 1) * dp_cols + (row - 1);
        for lane in 0..W {
            let s = base_s + lane;
            let code = nt_strand_code(seq[s + row - 1]);
            let prof_val = flat_profile[code * prof_len + (row - 1)];
            dp_buf[cell_idx * W + lane] = dp_buf[prev_cell_idx * W + lane] + prof_val;
        }
    }

    // Fill rest of band — inner loop over lanes is auto-vectorizable.
    for i in 1..=seq_len {
        let jmax = (i + max_gaps).min(prof_len);

        // Precompute profile values for each lane at row i.
        // codes[lane] = nt_code of seq byte at position base_s + lane + i - 1.
        let mut codes = [0usize; 8]; // padded for alignment
        for lane in 0..W {
            codes[lane] = nt_strand_code(seq[base_s + lane + i - 1]);
        }

        for j in (i + 1)..=jmax {
            let gap_val = gap_profile[j - 1];
            let diag_cell = ((i - 1) * dp_cols + (j - 1)) * W;
            let horiz_cell = (i * dp_cols + (j - 1)) * W;
            let dest_cell = (i * dp_cols + j) * W;

            for lane in 0..W {
                let prof_val = flat_profile[codes[lane] * prof_len + (j - 1)];
                let diag = dp_buf[diag_cell + lane] + prof_val;
                let horiz = dp_buf[horiz_cell + lane] + gap_val;
                dp_buf[dest_cell + lane] = diag.max(horiz);
            }
        }
    }

    // Extract scores.
    for lane in 0..W {
        for k in 0..num_gaps {
            let seq_consumed = min_len + k;
            scores_buf[lane * num_gaps + k] = if seq_consumed <= seq_len {
                let cell_idx = seq_consumed * dp_cols + prof_len;
                dp_buf[cell_idx * W + lane]
            } else {
                (LOG_ZERO * prof_len as f64) as f32
            };
        }
    }
}

/// Scalar DP fallback for remainder positions.
fn strand_dp_scalar(
    seq: &[u8],
    seq_len: usize,
    prof_len: usize,
    max_gaps: usize,
    min_len: usize,
    flat_profile: &[f64],
    gap_profile: &[f64],
    align: &mut [f64],
    dp_cols: usize,
    scores: &mut [f64],
) {
    align[0] = 0.0;
    for j in 1..=max_gaps.min(prof_len) {
        align[j] = align[j - 1] + gap_profile[j - 1];
    }
    for i in 1..=seq_len {
        let jmin = i;
        let jmax = (i + max_gaps).min(prof_len);
        for j in jmin..=jmax {
            align[i * dp_cols + j] = f64::NEG_INFINITY;
        }
    }

    for j in 1..=seq_len {
        let code = nt_strand_code(seq[j - 1]);
        let prof_val = flat_profile[code * prof_len + (j - 1)];
        align[j * dp_cols + j] = align[(j - 1) * dp_cols + (j - 1)] + prof_val;
    }

    for i in 1..=seq_len {
        let jmax = (i + max_gaps).min(prof_len);
        let code = nt_strand_code(seq[i - 1]);
        for j in (i + 1)..=jmax {
            let prof_val = flat_profile[code * prof_len + (j - 1)];
            let diag = align[(i - 1) * dp_cols + (j - 1)] + prof_val;
            let horiz = align[i * dp_cols + (j - 1)] + gap_profile[j - 1];
            align[i * dp_cols + j] = diag.max(horiz);
        }
    }

    for k in 0..=max_gaps {
        let seq_consumed = min_len + k;
        if seq_consumed <= seq_len {
            scores[k] = align[seq_consumed * dp_cols + prof_len];
        } else {
            scores[k] = LOG_ZERO * prof_len as f64;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;
    use crate::pattern;
    use crate::profile::Background;
    use crate::scoring;
    use crate::types::Region;

    fn load_test_pattern() -> (TrainingSet, Pattern) {
        let ts = epn::parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
        let region = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &region, &bg, 0.0002, -20.0);
        (ts, pat)
    }

    fn gen_seq(len: usize, seed: u64) -> Vec<u8> {
        let bases = [b'A', b'T', b'G', b'C'];
        let mut state = seed;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                bases[(state & 3) as usize]
            })
            .collect()
    }

    #[test]
    fn test_strand_nogap_matches_scalar() {
        let (_ts, pat) = load_test_pattern();
        let seq = gen_seq(5000, 42);

        for strand in &pat.strands {
            if strand.max_gaps > 0 {
                continue;
            }
            let scalar = scoring::compute_strand_score_table(&seq, strand);
            let simd = compute_strand_score_table_nogap(&seq, strand);

            assert_eq!(scalar[0].len(), simd.len(),
                "length mismatch for strand id={}", strand.id);
            for (i, (s, v)) in scalar[0].iter().zip(simd.iter()).enumerate() {
                assert!(
                    (s - v).abs() < 1e-4,
                    "strand id={} pos={}: scalar={} simd={}",
                    strand.id, i, s, v
                );
            }
        }
    }

    #[test]
    fn test_helix_matches_scalar() {
        let (_ts, pat) = load_test_pattern();
        let seq = gen_seq(5000, 42);

        for helix in &pat.helices {
            let scalar = scoring::compute_helix_score_table(&seq, helix);
            let simd = compute_helix_score_table(&seq, helix);

            assert_eq!(scalar.len(), simd.len(),
                "gap variant count mismatch for helix id={}", helix.id);
            for k in 0..scalar.len() {
                assert_eq!(scalar[k].len(), simd[k].len(),
                    "scan length mismatch for helix id={} gap={}", helix.id, k);
                for (i, (s, v)) in scalar[k].iter().zip(simd[k].iter()).enumerate() {
                    assert!(
                        (s - v).abs() < 1e-4,
                        "helix id={} gap={} pos={}: scalar={} simd={}",
                        helix.id, k, i, s, v
                    );
                }
            }
        }
    }

    #[test]
    fn test_strand_gapped_matches_scalar() {
        let (_ts, pat) = load_test_pattern();
        let seq = gen_seq(5000, 42);

        for strand in &pat.strands {
            if strand.max_gaps == 0 {
                continue;
            }
            let scalar = scoring::compute_strand_score_table(&seq, strand);
            let simd = compute_strand_score_table_gapped(&seq, strand);

            assert_eq!(scalar.len(), simd.len(),
                "gap variant count mismatch for strand id={}", strand.id);
            for k in 0..scalar.len() {
                assert_eq!(scalar[k].len(), simd[k].len(),
                    "scan length mismatch for strand id={} gap={}", strand.id, k);
                for (i, (s, v)) in scalar[k].iter().zip(simd[k].iter()).enumerate() {
                    assert!(
                        (s - v).abs() < 1e-3,
                        "strand id={} gap={} pos={}: scalar={} simd={}",
                        strand.id, k, i, s, v
                    );
                }
            }
        }
    }
}
