//! SIMD-accelerated scoring routines.
//!
//! These functions compute the same results as the scalar versions in scoring.rs
//! but use loop reordering and SIMD intrinsics for throughput.
//!
//! Strategy: process multiple scan positions simultaneously by reordering loops
//! from position-major (for s { for j }) to column-major (for j { for s }).

use std::cell::RefCell;

use crate::profile::{self, nt_strand_code, LOG_ZERO};
use crate::types::*;

thread_local! {
    /// Per-thread interleaved DP buffer for `compute_strand_score_table_gapped_into`.
    /// Reused across calls so the gapped-strand DP doesn't allocate per
    /// (scan-window × strand) — for the trna fixture this is dozens to hundreds
    /// of allocations per chunk.
    static SCRATCH_DP_BUF: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Per-thread per-lane scores buffer (W * num_gaps f32).
    static SCRATCH_SCORES: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    /// Per-thread scalar DP fallback buffers (f64 align + f64 scores).
    static SCRATCH_ALIGN_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_GAP_PROF_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_FLAT_PROF_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_SCORES_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    /// Per-thread helix-class encoded sequence (size scan_len + max_len - 1).
    static SCRATCH_HELIX_CODES: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    /// Per-thread strand nibble-encoded sequence (size scan_len + max_len - 1).
    /// Pre-encoding once amortizes the LUT lookup across all (lane, i, j) DP
    /// inner-loop entries — `multi_dp_w` previously called `nt_strand_code` per
    /// (lane, i) which compounds across deep DP fills.
    static SCRATCH_STRAND_CODES: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Set `buf` to length `len` with possibly-stale contents past its current
/// initialized region; callers must overwrite every position in `0..len`
/// before reading. `Vec::reserve(N)` ensures capacity ≥ `len() + N` (NOT
/// `capacity() + N`), so we compute `additional` relative to `buf.len()`.
#[inline]
fn scratch_uninit<T: Copy>(buf: &mut Vec<T>, len: usize) {
    if buf.capacity() < len {
        buf.reserve(len - buf.len());
    }
    // SAFETY: capacity ≥ len after the reserve above. Bytes beyond the previous
    // initialized region are stale; callers MUST overwrite every position
    // before reading.
    unsafe {
        buf.set_len(len);
    }
}

#[inline]
fn scratch_uninit_u8(buf: &mut Vec<u8>, len: usize) {
    if buf.capacity() < len {
        buf.reserve(len - buf.len());
    }
    unsafe {
        buf.set_len(len);
    }
}

/// Resize `table` to have exactly `outer_len` inner buffers, each with
/// `inner_len` accessible elements. Inner contents are uninitialized;
/// callers MUST overwrite every position they intend to read. Reuses the
/// existing `Vec<f32>` allocations when possible so per-thread scratch
/// buffers stay hot.
///
/// SAFETY: callers must write to every index in `0..inner_len` before reading
/// any element. The score table fns do this either by assigning on the first
/// profile column (helix, no-gap strand) or by row-by-row assignment (gapped
/// strand DP, multi_dp_w writes scores_buf into table[k][s] for every s).
#[inline]
fn shape_table_uninit(table: &mut Vec<Vec<f32>>, outer_len: usize, inner_len: usize) {
    table.truncate(outer_len);
    while table.len() < outer_len {
        table.push(Vec::new());
    }
    for inner in table.iter_mut() {
        inner.clear();
        inner.reserve(inner_len);
        // SAFETY: we just reserved `inner_len` capacity; bytes are
        // uninitialized but safe to set len since callers will overwrite.
        unsafe {
            inner.set_len(inner_len);
        }
    }
}

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
    let mut scores = Vec::new();
    compute_strand_score_table_nogap_into(seq, strand, &mut scores);
    scores
}

/// Same as [`compute_strand_score_table_nogap`] but writes into an existing
/// `Vec<f32>` so callers can pool the allocation across sequences.
pub fn compute_strand_score_table_nogap_into(seq: &[u8], strand: &Strand, scores: &mut Vec<f32>) {
    let max_len = strand.max_len;
    let scan_len = if seq.len() >= max_len {
        seq.len() - max_len + 1
    } else {
        scores.clear();
        return;
    };

    // Use the cached column-major profile (`profile_f32_t[j * 8 + code]`).
    // Building a per-call `col_vals` was redundant — the stage is identical
    // for every call against this Strand, so it now lives on the Strand and
    // gets reused across thousands of scan windows.
    let prof_t: &[f32] = &strand.profile_f32_t;

    scores.clear();
    scores.reserve(scan_len);
    // SAFETY: the j == 0 pass below assigns every position in 0..scan_len
    // before any reader looks at the buffer, so leaving the bytes
    // uninitialized briefly is fine.
    unsafe {
        scores.set_len(scan_len);
    }

    // Column-major: j == 0 assigns the running total (no zero-init pass
    // needed); j > 0 accumulates. Skipping the resize-with-zero saves one
    // full memory-bandwidth pass over the table.
    let chunks = scan_len / 8;
    let remainder = scan_len % 8;

    // j == 0: assignment.
    // SAFETY for both passes: prof_t.len() = max_len * 8, vals slice is 8 f32.
    // NT_NIBBLE_LUT returns a value in 0..=5, so `vals[code]` with vals.len() == 8
    // is in-bounds. seq_slice.len() == scan_len, scores.len() == scan_len.
    {
        let vals: &[f32] = &prof_t[0..8];
        let seq_slice = &seq[0..scan_len];
        for chunk in 0..chunks {
            let base = chunk * 8;
            for k in 0..8 {
                unsafe {
                    let code = NT_NIBBLE_LUT[(seq_slice.get_unchecked(base + k) & 0x0F) as usize];
                    *scores.get_unchecked_mut(base + k) =
                        *vals.get_unchecked(code as usize);
                }
            }
        }
        let base = chunks * 8;
        for k in 0..remainder {
            unsafe {
                let code = NT_NIBBLE_LUT[(seq_slice.get_unchecked(base + k) & 0x0F) as usize];
                *scores.get_unchecked_mut(base + k) =
                    *vals.get_unchecked(code as usize);
            }
        }
    }

    // j == 1..max_len: accumulation.
    for j in 1..max_len {
        let vals: &[f32] = &prof_t[j * 8..j * 8 + 8];
        let seq_slice = &seq[j..j + scan_len];
        for chunk in 0..chunks {
            let base = chunk * 8;
            for k in 0..8 {
                unsafe {
                    let code = NT_NIBBLE_LUT[(seq_slice.get_unchecked(base + k) & 0x0F) as usize];
                    *scores.get_unchecked_mut(base + k) +=
                        *vals.get_unchecked(code as usize);
                }
            }
        }
        let base = chunks * 8;
        for k in 0..remainder {
            unsafe {
                let code = NT_NIBBLE_LUT[(seq_slice.get_unchecked(base + k) & 0x0F) as usize];
                *scores.get_unchecked_mut(base + k) +=
                    *vals.get_unchecked(code as usize);
            }
        }
    }
}

/// Compute score table for a helix using column-major loop order.
///
/// For each gap variant k and profile column j, accumulates scores across all positions.
pub fn compute_helix_score_table(seq: &[u8], helix: &Helix) -> Vec<Vec<f32>> {
    let mut table = Vec::new();
    compute_helix_score_table_into(seq, helix, &mut table);
    table
}

/// Same as [`compute_helix_score_table`] but reuses an existing
/// `Vec<Vec<f32>>` so callers can pool the per-gap-variant inner buffers.
pub fn compute_helix_score_table_into(
    seq: &[u8],
    helix: &Helix,
    table: &mut Vec<Vec<f32>>,
) {
    let num_gaps = helix.max_gaps + 1;
    let scan_len = if seq.len() >= helix.max_len {
        seq.len() - helix.max_len + 1
    } else {
        shape_table_uninit(table, num_gaps, 0);
        return;
    };

    shape_table_uninit(table, num_gaps, scan_len);

    let hlen = helix.helix_len;

    // Use the cached per-column dinucleotide pair table. `pair[j*64 + (c5*8 +
    // c3)]` directly gives the profile score for the (c5, c3) helix class
    // pair at column j — folds the previous two-step lookup (c5, c3) →
    // DINUC_LUT → profile_f32_t into a single 64-byte cache-line read.
    let pair: &[f32] = &helix.pair_table;

    // Pre-encode the entire scan window into helix-class codes (0..6, see
    // `NT_HELIX_CLASS_LUT` in profile.rs). Without this, the inner loop calls
    // nt_helix_code() per (s, j, k), which re-runs the LUT for every base —
    // millions of redundant lookups for anything but tiny patterns. Use
    // per-thread scratch so chunked search reuses the same buffer instead of
    // allocating it per call.
    let needed = scan_len + helix.max_len - 1;

    SCRATCH_HELIX_CODES.with(|cell| {
        let mut codes_buf = cell.borrow_mut();
        scratch_uninit_u8(&mut codes_buf, needed);
        let codes = codes_buf.as_mut_slice();
        for (i, c) in codes.iter_mut().enumerate() {
            *c = profile::NT_HELIX_CLASS_LUT[seq[i] as usize];
        }

        let chunks = scan_len / 8;
        let remainder = scan_len % 8;

        for k in 0..num_gaps {
            let scores = &mut table[k];

            // j == 0: assignment pass (no zero-init pass needed).
            {
                let j = 0;
                // SAFETY: pair has length helix_len * 64, j < helix_len so
                // j*64 + 64 ≤ pair.len(). Slice of exactly 64 elements.
                let pair_col: &[f32] = &pair[j * 64..j * 64 + 64];
                let right_offset = helix.min_len - 1 + k - j;
                for chunk in 0..chunks {
                    let base = chunk * 8;
                    for i in 0..8 {
                        let s = base + i;
                        // SAFETY: codes has length scan_len + max_len - 1 ≥
                        // s + j and s + right_offset, with j ≤ helix_len-1
                        // and right_offset ≤ helix.max_len - 1, so both
                        // indices < codes.len(). c5,c3 ∈ [0..6] (per
                        // NT_HELIX_CLASS_LUT), so (c5 << 3) | c3 < 56 < 64
                        // = pair_col.len().
                        unsafe {
                            let c5 = *codes.get_unchecked(s + j) as usize;
                            let c3 = *codes.get_unchecked(s + right_offset) as usize;
                            *scores.get_unchecked_mut(s) =
                                *pair_col.get_unchecked((c5 << 3) | c3);
                        }
                    }
                }
                let base = chunks * 8;
                for i in 0..remainder {
                    let s = base + i;
                    unsafe {
                        let c5 = *codes.get_unchecked(s + j) as usize;
                        let c3 = *codes.get_unchecked(s + right_offset) as usize;
                        *scores.get_unchecked_mut(s) =
                            *pair_col.get_unchecked((c5 << 3) | c3);
                    }
                }
            }

            // j == 1..hlen: accumulation pass.
            for j in 1..hlen {
                let pair_col: &[f32] = &pair[j * 64..j * 64 + 64];
                let right_offset = helix.min_len - 1 + k - j;
                for chunk in 0..chunks {
                    let base = chunk * 8;
                    for i in 0..8 {
                        let s = base + i;
                        unsafe {
                            let c5 = *codes.get_unchecked(s + j) as usize;
                            let c3 = *codes.get_unchecked(s + right_offset) as usize;
                            *scores.get_unchecked_mut(s) +=
                                *pair_col.get_unchecked((c5 << 3) | c3);
                        }
                    }
                }
                let base = chunks * 8;
                for i in 0..remainder {
                    let s = base + i;
                    unsafe {
                        let c5 = *codes.get_unchecked(s + j) as usize;
                        let c3 = *codes.get_unchecked(s + right_offset) as usize;
                        *scores.get_unchecked_mut(s) +=
                            *pair_col.get_unchecked((c5 << 3) | c3);
                    }
                }
            }
        }
    });
}

/// Compute score table for a gapped strand using multi-position DP.
///
/// Runs W independent DP instances in parallel across scan positions.
/// Each "lane" processes a different scan position through the same DP structure.
pub fn compute_strand_score_table_gapped(seq: &[u8], strand: &Strand) -> Vec<Vec<f32>> {
    let mut table = Vec::new();
    compute_strand_score_table_gapped_into(seq, strand, &mut table);
    table
}

/// Same as [`compute_strand_score_table_gapped`] but writes into an existing
/// `Vec<Vec<f32>>` so callers can pool the per-gap-variant buffers.
pub fn compute_strand_score_table_gapped_into(
    seq: &[u8],
    strand: &Strand,
    table: &mut Vec<Vec<f32>>,
) {
    let max_len = strand.max_len;
    let max_gaps = strand.max_gaps;
    let min_len = strand.min_len;
    let num_gaps = max_gaps + 1;
    let scan_len = if seq.len() >= max_len {
        seq.len() - max_len + 1
    } else {
        shape_table_uninit(table, num_gaps, 0);
        return;
    };

    shape_table_uninit(table, num_gaps, scan_len);

    let prof_len = max_len;
    let seq_len = max_len;
    let dp_cols = prof_len + 1;

    // Use the column-major transposed profile (stride 8 = next power of two ≥
    // NT_CODES). Per-lane gather inside the DP inner loop now hits a single
    // 32-byte cache line instead of 4 lookups stridden by `prof_len * 4` bytes.
    let prof_t: &[f32] = &strand.profile_f32_t;
    let flat_profile: &[f32] = &strand.profile_f32;
    let gap_profile = &flat_profile[4 * prof_len..5 * prof_len];

    // Process W positions at a time. W=4 matches AVX1's xmm register width
    // for f32 ops; W=8 (ymm) was tried but didn't help measurably (compiler
    // already widened the inner lane loop using fma-style packing) and
    // doubled the dp_buf footprint.
    const W: usize = 4;

    let full_chunks = scan_len / W;
    let remainder = scan_len % W;

    // Pre-encode the strand's nibble codes for the entire scan window once,
    // so multi_dp_w doesn't redo `seq[..] -> nt_strand_code` per (lane, i).
    let codes_needed = scan_len + max_len - 1;

    SCRATCH_DP_BUF.with(|dp_cell| {
    SCRATCH_SCORES.with(|sc_cell| {
    SCRATCH_STRAND_CODES.with(|codes_cell| {
        let mut dp_buf = dp_cell.borrow_mut();
        let mut scores_buf = sc_cell.borrow_mut();
        let mut codes = codes_cell.borrow_mut();

        let dp_size = (seq_len + 1) * dp_cols;
        scratch_uninit(&mut dp_buf, W * dp_size);
        scratch_uninit(&mut scores_buf, W * num_gaps);
        scratch_uninit_u8(&mut codes, codes_needed);

        for (i, c) in codes.iter_mut().enumerate() {
            *c = NT_NIBBLE_LUT[(seq[i] & 0x0F) as usize];
        }

        for chunk in 0..full_chunks {
            let base_s = chunk * W;

            // Reset and fill W DP tables in parallel.
            multi_dp_w::<W>(
                &codes, base_s, seq_len, prof_len, max_gaps, min_len,
                prof_t, gap_profile, dp_cols,
                &mut dp_buf, &mut scores_buf,
            );

            // Write results.
            for lane in 0..W {
                for k in 0..num_gaps {
                    table[k][base_s + lane] = scores_buf[lane * num_gaps + k];
                }
            }
        }

        // Handle remainder positions one at a time using the scalar path.
        if remainder > 0 {
            SCRATCH_ALIGN_F64.with(|al_cell| {
            SCRATCH_FLAT_PROF_F64.with(|flat_cell| {
            SCRATCH_GAP_PROF_F64.with(|gap_cell| {
            SCRATCH_SCORES_F64.with(|sc64_cell| {
                let mut align = al_cell.borrow_mut();
                let mut flat64 = flat_cell.borrow_mut();
                let mut gap64 = gap_cell.borrow_mut();
                let mut sc = sc64_cell.borrow_mut();

                scratch_uninit(&mut align, dp_size);
                if flat64.len() != flat_profile.len() {
                    flat64.clear();
                    flat64.extend(flat_profile.iter().map(|&v| v as f64));
                } else {
                    for (dst, &src) in flat64.iter_mut().zip(flat_profile.iter()) {
                        *dst = src as f64;
                    }
                }
                if gap64.len() != gap_profile.len() {
                    gap64.clear();
                    gap64.extend(gap_profile.iter().map(|&v| v as f64));
                } else {
                    for (dst, &src) in gap64.iter_mut().zip(gap_profile.iter()) {
                        *dst = src as f64;
                    }
                }
                scratch_uninit(&mut sc, num_gaps);

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
            });
            });
            });
            });
        }
    });
    });
    });
}

/// Run W independent DP instances on consecutive scan positions.
///
/// Uses interleaved SOA layout: dp_buf[(i * dp_cols + j) * W + lane]
/// so that all W lanes for the same cell are adjacent in memory,
/// enabling the compiler to auto-vectorize the inner lane loop.
///
/// `seq_codes` is the entire scan window encoded once via NT_NIBBLE_LUT, so
/// per-lane lookups inside the DP fill become a single byte read (no LUT in
/// the hot loop). `prof_t[j * 8 + code]` is the column-major transposed
/// profile — the per-lane gather hits a 32-byte cache line on every column.
fn multi_dp_w<const W: usize>(
    seq_codes: &[u8],
    base_s: usize,
    seq_len: usize,
    prof_len: usize,
    max_gaps: usize,
    min_len: usize,
    prof_t: &[f32],
    gap_profile: &[f32],
    dp_cols: usize,
    dp_buf: &mut [f32],
    scores_buf: &mut [f32],
) {
    let num_gaps = max_gaps + 1;
    // Interleaved layout: cell(i, j) for lane L is at [(i * dp_cols + j) * W + L].
    //
    // We deliberately *don't* zero the full dp_buf here. The band cells written
    // below are (0, 0..=max_gaps), the diagonal (i, i) for i in 1..=seq_len,
    // and the band (i, j) with j in (i+1)..=min(i+max_gaps, prof_len). Every
    // cell that the inner loop reads is within that envelope and is written
    // before it is read (band writes proceed row-major; reads only touch
    // (i-1, j-1) and (i, j-1), both of which sit inside the previously-written
    // band). The score extraction at the end reads (seq_consumed, prof_len)
    // which is the band's upper edge — always written. So leaving stale values
    // outside the band is safe across W-position chunks. This drops one
    // (seq_len+1) * (prof_len+1) * W f32 fill per chunk.

    // align[0][0] = 0 for all lanes.
    for lane in 0..W {
        dp_buf[lane] = 0.0;
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
        let col_base = (row - 1) * 8;
        for lane in 0..W {
            let s = base_s + lane;
            let code = seq_codes[s + row - 1] as usize;
            let prof_val = prof_t[col_base + code];
            dp_buf[cell_idx * W + lane] = dp_buf[prev_cell_idx * W + lane] + prof_val;
        }
    }

    // Fill rest of band — inner loop over lanes is auto-vectorizable.
    for i in 1..=seq_len {
        let jmax = (i + max_gaps).min(prof_len);

        // Precompute the lane codes once per row i (same code for every j in
        // this row).
        let mut codes = [0u8; 8];
        for lane in 0..W {
            codes[lane] = seq_codes[base_s + lane + i - 1];
        }

        for j in (i + 1)..=jmax {
            let gap_val = gap_profile[j - 1];
            let diag_cell = ((i - 1) * dp_cols + (j - 1)) * W;
            let horiz_cell = (i * dp_cols + (j - 1)) * W;
            let dest_cell = (i * dp_cols + j) * W;
            let col_base = (j - 1) * 8;

            for lane in 0..W {
                // Per-lane gather into one 32-byte row of prof_t — all 4 lanes
                // hit the same cache line. Strided gather into the row-major
                // `flat_profile[code * prof_len + j]` was a much wider stride
                // (prof_len * 4 bytes per lane).
                let prof_val = prof_t[col_base + codes[lane] as usize];
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
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
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
