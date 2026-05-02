use std::cell::RefCell;

use crate::profile::{nt_helix_code, nt_strand_code};
use crate::simd;
use crate::types::*;

thread_local! {
    /// Per-thread scratch for `score_strand_run_dp`'s DP buffer (`align`) and
    /// pre-encoded query codes. The DP path is called once per
    /// (variable strand × DMP hit candidate); pooling these avoids two heap
    /// allocations per call. The buffer can shrink between calls but never
    /// frees its capacity.
    static SCRATCH_DP_ALIGN_F64: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_DP_CODES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    /// Per-thread scratch for `score_configs_direct_with`'s per-mask working
    /// vectors (st_fixed/hx_fixed/upper-bound/suffix-max). The DMP path calls
    /// score_configs_direct_with once per surviving level-1+ candidate; on
    /// large IB intron scans this can be hundreds of calls and the small
    /// per-call Vec allocations add up.
    static SCRATCH_FIXED: RefCell<Vec<bool>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_UPPER: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
    static SCRATCH_SUFFIX: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

#[inline]
fn ensure_len_f64(buf: &mut Vec<f64>, len: usize, fill: f64) {
    if buf.len() < len {
        buf.resize(len, fill);
    } else {
        buf.truncate(len);
    }
}

/// Precomputed score tables for all elements in a mask.
pub struct ScoreTables {
    /// Score tables for each helix in the mask.
    /// `helix_scores[i][gap_variant][position]`
    pub helix_scores: Vec<Vec<Vec<f32>>>,
    /// Score tables for each strand in the mask.
    /// `strand_scores[i][gap_variant][position]`
    pub strand_scores: Vec<Vec<Vec<f32>>>,
}

/// Compute score table for a strand across all positions in a sequence.
///
/// Returns `scores[gap_variant][position]` where position ranges from 0 to
/// `seq_len - strand.max_len`.
pub fn compute_strand_score_table(seq: &[u8], strand: &Strand) -> Vec<Vec<f32>> {
    if strand.max_gaps == 0 {
        let scores = simd::compute_strand_score_table_nogap(seq, strand);
        if scores.is_empty() {
            vec![vec![]; 1]
        } else {
            vec![scores]
        }
    } else {
        simd::compute_strand_score_table_gapped(seq, strand)
    }
}

/// Compute score table for a helix across all positions in a sequence.
///
/// Returns `scores[gap_variant][position]` where gap_variant ranges from
/// 0 to helix.max_gaps and position from 0 to `seq_len - helix.max_len`.
pub fn compute_helix_score_table(seq: &[u8], helix: &Helix) -> Vec<Vec<f32>> {
    simd::compute_helix_score_table(seq, helix)
}

/// Compute score tables for all elements in a mask.
pub fn compute_mask_score_tables(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
) -> ScoreTables {
    let mut tables = ScoreTables {
        helix_scores: Vec::new(),
        strand_scores: Vec::new(),
    };
    compute_mask_score_tables_into(seq, pattern, mask, &mut tables);
    tables
}

/// Same as [`compute_mask_score_tables`] but reuses an existing `ScoreTables`
/// so the `Vec<f32>` per-element/per-gap-variant buffers stay hot across
/// many calls. Pair with `thread_local!` scratch storage for big wins on
/// parallel multi-sequence searches.
pub fn compute_mask_score_tables_into(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
    tables: &mut ScoreTables,
) {
    let nh = mask.hx_indices.len();
    let ns = mask.st_indices.len();
    tables.helix_scores.truncate(nh);
    while tables.helix_scores.len() < nh {
        tables.helix_scores.push(Vec::new());
    }
    tables.strand_scores.truncate(ns);
    while tables.strand_scores.len() < ns {
        tables.strand_scores.push(Vec::new());
    }

    for (i, &idx) in mask.hx_indices.iter().enumerate() {
        simd::compute_helix_score_table_into(
            seq,
            &pattern.helices[idx],
            &mut tables.helix_scores[i],
        );
    }

    for (i, &idx) in mask.st_indices.iter().enumerate() {
        let strand = &pattern.strands[idx];
        let dst = &mut tables.strand_scores[i];
        if strand.max_gaps == 0 {
            dst.truncate(1);
            while dst.len() < 1 {
                dst.push(Vec::new());
            }
            simd::compute_strand_score_table_nogap_into(seq, strand, &mut dst[0]);
        } else {
            simd::compute_strand_score_table_gapped_into(seq, strand, dst);
        }
    }
}

/// A precomputed lookup entry for fast config scoring. Each "entry" is a
/// raw `*const f32` pointing into one of the score tables, already offset by
/// the config's per-element `bgn`, so the inner loop is a single pointer
/// add + load. The slice length is folded into `max_safe_tab` so we don't
/// carry it through the hot path.
///
/// SAFETY: pointers reference data inside `ScoreTables` (the `'a` lifetime
/// parameter); the lookup must not outlive the tables it was built from.
pub struct ConfigLookup<'a> {
    /// Flat array of pointers, length `n_configs * n_elements`. Layout:
    /// `entries[cfg_idx * n_elements + ei]`. Empty / invalid configs have
    /// their stride filled with null but `max_safe_tab[cfg_idx] == 0`, so
    /// the inner loop never dereferences them.
    entries: Vec<*const f32>,
    /// Number of elements per config (mask.st_indices.len() + mask.hx_indices.len()).
    n_elements: usize,
    /// Config lengths.
    config_lens: Vec<usize>,
    /// Suffix max scores: suffix_max[i] = max score achievable from elements i..n.
    /// Used for early termination.
    suffix_max: Vec<f64>,
    /// Per-config: smallest `tab_index` value at which any element falls off
    /// the end of its score slice. Equivalently, `min over ei of (slice.len()
    /// - bgn)`. The inner loop checks `tab_index < max_safe_tab[cfg]` once
    /// and then dereferences without bounds checks.
    max_safe_tab: Vec<usize>,
    /// Anchor lifetime to the tables we built from.
    _phantom: std::marker::PhantomData<&'a ()>,
}

// SAFETY: ConfigLookup contains raw pointers to data owned elsewhere. As
// long as the lifetime contract is upheld (`tables` outlives the lookup),
// it's safe to share across threads — the pointed-to data is read-only.
unsafe impl<'a> Send for ConfigLookup<'a> {}
unsafe impl<'a> Sync for ConfigLookup<'a> {}

impl<'a> ConfigLookup<'a> {
    /// Build a precomputed config lookup from mask and score tables.
    pub fn build(mask: &ResolvedMask, tables: &'a ScoreTables) -> Self {
        let n_elements = mask.st_indices.len() + mask.hx_indices.len();
        let n_configs = mask.configs.len();
        let mut config_lens = Vec::with_capacity(n_configs);

        // Compute per-element global max score (for early termination bound).
        let mut element_max = vec![f32::NEG_INFINITY; n_elements];
        for (s, scores) in tables.strand_scores.iter().enumerate() {
            for gap_scores in scores.iter() {
                for &v in gap_scores.iter() {
                    element_max[s] = element_max[s].max(v);
                }
            }
        }
        let st_count = tables.strand_scores.len();
        for (h, scores) in tables.helix_scores.iter().enumerate() {
            for gap_scores in scores.iter() {
                for &v in gap_scores.iter() {
                    element_max[st_count + h] = element_max[st_count + h].max(v);
                }
            }
        }

        // Build suffix max: suffix_max[i] = sum of element_max[i..n].
        let mut suffix_max = vec![0.0f64; n_elements + 1];
        for i in (0..n_elements).rev() {
            suffix_max[i] = suffix_max[i + 1] + element_max[i] as f64;
        }

        // Flat entries array: stride n_elements per config.
        let mut entries: Vec<*const f32> = vec![std::ptr::null(); n_configs * n_elements];
        let mut max_safe_tab: Vec<usize> = vec![0; n_configs];

        let nst = mask.st_indices.len();
        let nhx = mask.hx_indices.len();

        for (cfg_idx, cfg) in mask.configs.iter().enumerate() {
            let row = cfg_idx * n_elements;
            let mut valid = true;
            let mut safe = usize::MAX;

            for s in 0..nst {
                let gap = cfg.st_gaps[s];
                if gap >= tables.strand_scores[s].len() {
                    valid = false;
                    break;
                }
                let slice = tables.strand_scores[s][gap].as_slice();
                let bgn = cfg.st_bgn[s];
                if bgn > slice.len() {
                    valid = false;
                    break;
                }
                // SAFETY: bgn ≤ slice.len(); the resulting pointer points one
                // past the slice's last element at worst, which is allowed.
                entries[row + s] = unsafe { slice.as_ptr().add(bgn) };
                let safe_for_e = slice.len() - bgn;
                if safe_for_e < safe {
                    safe = safe_for_e;
                }
            }

            if valid {
                for h in 0..nhx {
                    let gap = cfg.hx_gaps[h];
                    if gap >= tables.helix_scores[h].len() {
                        valid = false;
                        break;
                    }
                    let slice = tables.helix_scores[h][gap].as_slice();
                    let bgn = cfg.hx_bgn[h];
                    if bgn > slice.len() {
                        valid = false;
                        break;
                    }
                    entries[row + nst + h] = unsafe { slice.as_ptr().add(bgn) };
                    let safe_for_e = slice.len() - bgn;
                    if safe_for_e < safe {
                        safe = safe_for_e;
                    }
                }
            }

            // Invalid configs end up with safe=0 (or original usize::MAX if
            // n_elements==0). In either case the inner loop's `tab_index >=
            // safe` check skips them.
            if !valid || n_elements == 0 {
                max_safe_tab[cfg_idx] = 0;
            } else {
                max_safe_tab[cfg_idx] = safe;
            }
            config_lens.push(cfg.len);
        }

        Self {
            entries,
            n_elements,
            config_lens,
            suffix_max,
            max_safe_tab,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Find the best-scoring configuration at a given position.
    #[inline]
    pub fn best_score(&self, tab_index: usize) -> (f64, usize) {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        let n_elements = self.n_elements;

        for cfg_idx in 0..self.config_lens.len() {
            // Per-config bounds check (replaces the per-element `pos <
            // slice.len()` test in the inner loop). Invalid configs have
            // max_safe_tab=0 and are filtered here.
            if tab_index >= unsafe { *self.max_safe_tab.get_unchecked(cfg_idx) } {
                continue;
            }
            let row = cfg_idx * n_elements;
            let mut score = 0.0f64;
            let mut early_exit = false;

            for ei in 0..n_elements {
                // SAFETY: tab_index < max_safe_tab[cfg_idx] ≤ slice.len() -
                // bgn, so the offset pointer stored at `entries[row + ei]`
                // plus `tab_index` lands inside the slice.
                let ptr = unsafe { *self.entries.get_unchecked(row + ei) };
                score += unsafe { *ptr.add(tab_index) } as f64;
                if score + unsafe { *self.suffix_max.get_unchecked(ei + 1) } <= best_score {
                    early_exit = true;
                    break;
                }
            }

            if !early_exit && score > best_score {
                best_score = score;
                best_idx = cfg_idx;
            }
        }

        (best_score, best_idx)
    }

    /// Find the best-scoring configuration, with threshold-based pruning.
    /// Skips positions where even the upper bound can't reach the threshold.
    #[inline]
    pub fn best_score_threshold(&self, tab_index: usize, threshold: f64) -> (f64, usize) {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;
        let n_elements = self.n_elements;

        for cfg_idx in 0..self.config_lens.len() {
            if tab_index >= unsafe { *self.max_safe_tab.get_unchecked(cfg_idx) } {
                continue;
            }
            let row = cfg_idx * n_elements;
            let mut score = 0.0f64;
            let mut early_exit = false;
            // best_score only changes between configs, so the pruning target
            // is invariant within the inner loop. Hoist the branch.
            let min_target = if best_score > threshold { best_score } else { threshold };

            for ei in 0..n_elements {
                // SAFETY: see best_score above.
                let ptr = unsafe { *self.entries.get_unchecked(row + ei) };
                score += unsafe { *ptr.add(tab_index) } as f64;
                if score + unsafe { *self.suffix_max.get_unchecked(ei + 1) } <= min_target {
                    early_exit = true;
                    break;
                }
            }

            if !early_exit && score > best_score {
                best_score = score;
                best_idx = cfg_idx;
            }
        }

        (best_score, best_idx)
    }

    /// Get the config length for a given config index.
    #[inline]
    pub fn config_len(&self, idx: usize) -> usize {
        self.config_lens[idx]
    }
}

/// Find the best-scoring configuration at a given position.
///
/// Returns (best_score, best_config_index).
pub fn best_config_score(
    mask: &ResolvedMask,
    tables: &ScoreTables,
    tab_index: usize,
) -> (f64, usize) {
    let mut best_score = f64::NEG_INFINITY;
    let mut best_idx = 0;

    for (cfg_idx, cfg) in mask.configs.iter().enumerate() {
        let mut score = 0.0f64;

        // Sum strand scores.
        for s in 0..mask.st_indices.len() {
            let gap = cfg.st_gaps[s];
            let pos = tab_index + cfg.st_bgn[s];
            if gap < tables.strand_scores[s].len() && pos < tables.strand_scores[s][gap].len() {
                score += tables.strand_scores[s][gap][pos] as f64;
            } else {
                score = f64::NEG_INFINITY;
                break;
            }
        }

        // Sum helix scores.
        if score > f64::NEG_INFINITY {
            for h in 0..mask.hx_indices.len() {
                let gap = cfg.hx_gaps[h];
                let pos = tab_index + cfg.hx_bgn[h];
                if gap < tables.helix_scores[h].len()
                    && pos < tables.helix_scores[h][gap].len()
                {
                    score += tables.helix_scores[h][gap][pos] as f64;
                } else {
                    score = f64::NEG_INFINITY;
                    break;
                }
            }
        }

        if score > best_score {
            best_score = score;
            best_idx = cfg_idx;
        }
    }

    (best_score, best_idx)
}

/// Score configs directly against sequence data at a given position (DMP).
///
/// This avoids precomputing score tables for all gap variants, which is
/// prohibitively expensive when helices have large distance ranges.
///
/// Optimization: identifies elements whose (position, gap) is identical
/// across all configs ("fixed") and scores them once. Only elements that
/// vary across configs are re-scored per-config.
///
/// Returns (best_score, best_config_index).
pub fn score_configs_direct(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
    seq_pos: usize,
) -> (f64, usize) {
    score_configs_direct_with(seq, pattern, mask, &mask.configs, seq_pos, mask.threshold)
}

/// Variant of `score_configs_direct` that accepts an external configs slice
/// (used by the DMP path, which produces per-hit constrained configs without
/// allocating a fresh `ResolvedMask`). `threshold` is the score floor used for
/// pruning.
pub fn score_configs_direct_with(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
    configs: &[Config],
    seq_pos: usize,
    threshold: f64,
) -> (f64, usize) {
    if configs.is_empty() {
        return (f64::NEG_INFINITY, 0);
    }

    // Single-config DMP fast path: every element is trivially fixed; one
    // sweep covers it.
    if configs.len() == 1 {
        let cfg = &configs[0];
        if seq_pos + cfg.len > seq.len() {
            return (f64::NEG_INFINITY, 0);
        }
        let mut score = 0.0f64;
        for (s, &st_idx) in mask.st_indices.iter().enumerate() {
            let strand = &pattern.strands[st_idx];
            let pos = seq_pos + cfg.st_bgn[s];
            let len = strand.min_len + cfg.st_gaps[s];
            if pos + len > seq.len() {
                return (f64::NEG_INFINITY, 0);
            }
            score += score_strand_run(strand, &seq[pos..pos + len]);
        }
        for (h, &hx_idx) in mask.hx_indices.iter().enumerate() {
            let helix = &pattern.helices[hx_idx];
            let pos5 = seq_pos + cfg.hx_bgn[h];
            let dist = helix.min_dist + cfg.hx_gaps[h];
            let pos3 = pos5 + helix.helix_len + dist;
            if pos3 + helix.helix_len > seq.len() {
                return (f64::NEG_INFINITY, 0);
            }
            score += score_helix_run(
                helix,
                &seq[pos5..pos5 + helix.helix_len],
                &seq[pos3..pos3 + helix.helix_len],
            );
        }
        return (score, 0);
    }

    let ref0 = &configs[0];
    let nst = mask.st_indices.len();
    let nhx = mask.hx_indices.len();
    let n_total = nst + nhx;

    SCRATCH_FIXED.with(|fx_cell| {
    SCRATCH_UPPER.with(|up_cell| {
    SCRATCH_SUFFIX.with(|sf_cell| {
        let mut fixed_buf = fx_cell.borrow_mut();
        let mut upper_buf = up_cell.borrow_mut();
        let mut suffix_buf = sf_cell.borrow_mut();

        // fixed_buf layout: [st_fixed (nst); hx_fixed (nhx)]. upper_buf
        // layout: [st_upper (nst); hx_upper (nhx)]. suffix_buf: n_total + 1.
        if fixed_buf.len() < n_total {
            fixed_buf.resize(n_total, false);
        } else {
            fixed_buf.truncate(n_total);
        }
        if upper_buf.len() < n_total {
            upper_buf.resize(n_total, 0.0);
        } else {
            upper_buf.truncate(n_total);
        }
        if suffix_buf.len() < n_total + 1 {
            suffix_buf.resize(n_total + 1, 0.0);
        } else {
            suffix_buf.truncate(n_total + 1);
        }

        // Identify fixed strands and helices.
        for s in 0..nst {
            fixed_buf[s] = configs.iter().all(|c| {
                c.st_bgn[s] == ref0.st_bgn[s] && c.st_gaps[s] == ref0.st_gaps[s]
            });
        }
        for h in 0..nhx {
            fixed_buf[nst + h] = configs.iter().all(|c| {
                c.hx_bgn[h] == ref0.hx_bgn[h] && c.hx_gaps[h] == ref0.hx_gaps[h]
            });
        }

        // Score fixed elements once.
        let mut fixed_score = 0.0f64;
        let mut fixed_valid = true;

        for (s, &st_idx) in mask.st_indices.iter().enumerate() {
            if !fixed_buf[s] {
                continue;
            }
            let strand = &pattern.strands[st_idx];
            let pos = seq_pos + ref0.st_bgn[s];
            let len = strand.min_len + ref0.st_gaps[s];
            if pos + len > seq.len() {
                fixed_valid = false;
                break;
            }
            fixed_score += score_strand_run(strand, &seq[pos..pos + len]);
        }

        if fixed_valid {
            for (h, &hx_idx) in mask.hx_indices.iter().enumerate() {
                if !fixed_buf[nst + h] {
                    continue;
                }
                let helix = &pattern.helices[hx_idx];
                let pos5 = seq_pos + ref0.hx_bgn[h];
                let dist = helix.min_dist + ref0.hx_gaps[h];
                let pos3 = pos5 + helix.helix_len + dist;
                if pos3 + helix.helix_len > seq.len() {
                    fixed_valid = false;
                    break;
                }
                fixed_score += score_helix_run(
                    helix,
                    &seq[pos5..pos5 + helix.helix_len],
                    &seq[pos3..pos3 + helix.helix_len],
                );
            }
        }

        if !fixed_valid {
            return (f64::NEG_INFINITY, 0);
        }

        // Per-variable-element upper bounds.
        for (s, &idx) in mask.st_indices.iter().enumerate() {
            upper_buf[s] = if fixed_buf[s] {
                0.0
            } else {
                pattern.strands[idx].upper_bound
            };
        }
        for (h, &idx) in mask.hx_indices.iter().enumerate() {
            upper_buf[nst + h] = if fixed_buf[nst + h] {
                0.0
            } else {
                pattern.helices[idx].upper_bound
            };
        }

        // suffix_max[i] = sum of upper bounds for all variable elements at
        // positions ≥ i (single backward sweep over concatenated st+hx bounds).
        suffix_buf[n_total] = 0.0;
        for i in (0..n_total).rev() {
            suffix_buf[i] = suffix_buf[i + 1] + upper_buf[i];
        }

        // Score variable elements per-config.
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (cfg_idx, cfg) in configs.iter().enumerate() {
            if seq_pos + cfg.len > seq.len() {
                continue;
            }

            let mut score = fixed_score;
            let mut valid = true;
            let target = best_score.max(threshold);

            for (s, &st_idx) in mask.st_indices.iter().enumerate() {
                if fixed_buf[s] {
                    continue;
                }
                if score + suffix_buf[s] <= target {
                    valid = false;
                    break;
                }
                let strand = &pattern.strands[st_idx];
                let pos = seq_pos + cfg.st_bgn[s];
                let len = strand.min_len + cfg.st_gaps[s];
                if pos + len > seq.len() {
                    valid = false;
                    break;
                }
                score += score_strand_run(strand, &seq[pos..pos + len]);
            }

            if valid {
                for (h, &hx_idx) in mask.hx_indices.iter().enumerate() {
                    if fixed_buf[nst + h] {
                        continue;
                    }
                    if score + suffix_buf[nst + h] <= target {
                        valid = false;
                        break;
                    }
                    let helix = &pattern.helices[hx_idx];
                    let pos5 = seq_pos + cfg.hx_bgn[h];
                    let dist = helix.min_dist + cfg.hx_gaps[h];
                    let pos3 = pos5 + helix.helix_len + dist;
                    if pos3 + helix.helix_len > seq.len() {
                        valid = false;
                        break;
                    }
                    score += score_helix_run(
                        helix,
                        &seq[pos5..pos5 + helix.helix_len],
                        &seq[pos3..pos3 + helix.helix_len],
                    );
                }
            }

            if valid && score > best_score {
                best_score = score;
                best_idx = cfg_idx;
            }
        }

        (best_score, best_idx)
    })
    })
    })
}

/// Score a contiguous run of `seq.len()` bases against the leading columns
/// of `strand.profile_f32`. For ungapped strands (`min_len == max_len`),
/// this is a straight sum over the diagonal (one profile column per base).
/// For gapped strands (`max_gaps > 0`), the alignment is found via a banded
/// DP that mirrors C ERPIN's `AlignSProfile` (libsrc/align.c): the DP
/// returns `Align[seq.len()][max_len]` — the best score aligning `seq.len()`
/// query bases to all `max_len` profile columns, with `max_len - seq.len()`
/// gap insertions distributed optimally. The straight diagonal sum is
/// suboptimal in general; using it for gapped strands underscores the
/// alignment by ~20 points on real intron sequences and was responsible
/// for missing C parity hits on the IB intron test case.
#[inline]
fn score_strand_run(strand: &Strand, seq: &[u8]) -> f64 {
    if strand.max_gaps == 0 {
        return score_strand_run_linear(strand, seq);
    }
    score_strand_run_dp(strand, seq)
}

#[inline]
fn score_strand_run_linear(strand: &Strand, seq: &[u8]) -> f64 {
    let cols = strand.max_len;
    let prof = strand.profile_f32.as_slice();
    let mut acc = 0.0f64;
    for (j, &b) in seq.iter().enumerate() {
        let code = nt_strand_code(b);
        // SAFETY: `code < 6` (per `nt_strand_code` LUT) and `j < seq.len() <=
        // strand.max_len = cols`, so `code * cols + j < 6 * cols == prof.len()`.
        acc += unsafe { *prof.get_unchecked(code * cols + j) } as f64;
    }
    acc
}

/// Banded gapped-strand DP. Mirrors C ERPIN's `AlignSProfile` (libsrc/align.c).
/// Returns the score for aligning `seq.len()` query bases to `strand.max_len`
/// profile columns, with `max_len - seq.len()` insertions (= gap row, code 4
/// in the profile) distributed optimally along the alignment.
///
/// Required: `min_len <= seq.len() <= max_len`.
#[inline]
fn score_strand_run_dp(strand: &Strand, seq: &[u8]) -> f64 {
    let m = seq.len();
    let n = strand.max_len;
    let max_gaps = strand.max_gaps;
    let prof = strand.profile_f32.as_slice();
    let stride = n + 1;

    SCRATCH_DP_ALIGN_F64.with(|al_cell| {
        SCRATCH_DP_CODES.with(|cd_cell| {
            let mut align = al_cell.borrow_mut();
            let mut codes = cd_cell.borrow_mut();

            // Pre-encode query bases. Reuse codes buffer.
            ensure_len_usize(&mut codes, m);
            for i in 0..m {
                codes[i] = nt_strand_code(seq[i]);
            }

            // align[i][j]: best score aligning first i query bases to first j
            // profile cols. Band constraint: 0 <= j - i <= max_gaps. Out-of-band
            // cells stay NEG_INFINITY and are never read by the in-band
            // recurrence. Initialize the whole buffer to NEG_INFINITY only on
            // grow; on reuse we overwrite the band cells before reading them
            // and the out-of-band cells must remain NEG_INFINITY (in case a
            // prior call's max_gaps was wider than this call's). Easiest: just
            // re-fill the band region.
            let needed = (m + 1) * stride;
            ensure_len_f64(&mut align, needed, f64::NEG_INFINITY);
            // The recurrence reads (i-1, j-1) for diag and (i, j-1) for horiz.
            // Both writes are inside the band [i, i+max_gaps], so any
            // out-of-band leftovers from a prior call must stay NEG_INFINITY
            // OR we must overwrite with NEG_INFINITY in the band region.
            // Since we may have shrunk from a wider band before, force-reset
            // the band cells to NEG_INFINITY and write the in-band values
            // explicitly. Easier and correct: zero (set NEG_INFINITY) the
            // entire `needed` region. Costs O(m * n) — but m, n are tiny
            // (single-digit) so this is cheap relative to the DP.
            for v in align.iter_mut().take(needed) {
                *v = f64::NEG_INFINITY;
            }
            align[0] = 0.0;

            // Row 0 (pure-insertion path).
            let prof_x_offset = 4 * n;
            for j in 1..=max_gaps.min(n) {
                align[j] = align[j - 1] + prof[prof_x_offset + j - 1] as f64;
            }

            // Diagonal.
            let limit_diag = m.min(n);
            for j in 1..=limit_diag {
                let code = codes[j - 1];
                let v = prof[code * n + j - 1] as f64;
                align[j * stride + j] = align[(j - 1) * stride + (j - 1)] + v;
            }

            // In-band fill.
            for i in 1..=m {
                let jmax = (i + max_gaps).min(n);
                let code = codes[i - 1];
                let prof_row_off = code * n;
                for j in (i + 1)..=jmax {
                    let jm1 = j - 1;
                    let match_score =
                        align[(i - 1) * stride + jm1] + prof[prof_row_off + jm1] as f64;
                    let gap_score =
                        align[i * stride + jm1] + prof[prof_x_offset + jm1] as f64;
                    align[i * stride + j] = match_score.max(gap_score);
                }
            }

            align[m * stride + n]
        })
    })
}

#[inline]
fn ensure_len_usize(buf: &mut Vec<usize>, len: usize) {
    if buf.len() < len {
        buf.resize(len, 0);
    } else {
        buf.truncate(len);
    }
}

/// Score a helix's two strands of length `helix.helix_len` each. The 3'
/// strand is read in reverse (Watson-Crick pairing reads the complement of
/// the 3' nucleotide farthest from the 5' position).
#[inline]
fn score_helix_run(helix: &Helix, seq5: &[u8], seq3: &[u8]) -> f64 {
    let cols = helix.helix_len;
    debug_assert_eq!(seq5.len(), cols);
    debug_assert_eq!(seq3.len(), cols);
    let prof = helix.profile_f32.as_slice();
    let mut acc = 0.0f64;
    for j in 0..cols {
        let code = nt_helix_code(seq5[j], seq3[cols - 1 - j]);
        // SAFETY: `code < DINUC_CODES (= 26)` so `code * cols + j < 26 * cols
        // == prof.len()`.
        acc += unsafe { *prof.get_unchecked(code * cols + j) } as f64;
    }
    acc
}

/// Score a single training set sequence against a mask. Used for threshold
/// computation (one-time per mask × training-sequence at search startup).
pub fn score_training_sequence(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
) -> f64 {
    let mut total_score = 0.0;

    for &st_idx in &mask.st_indices {
        let strand = &pattern.strands[st_idx];
        let cols = strand.max_len;
        let prof = strand.profile_f32.as_slice();
        let mut score = 0.0f64;
        for j in 0..cols {
            let code = nt_strand_code(seq[strand.db_begin + j]);
            score += prof[code * cols + j] as f64;
        }
        total_score += score;
    }

    for &hx_idx in &mask.hx_indices {
        let helix = &pattern.helices[hx_idx];
        let cols = helix.helix_len;
        let prof = helix.profile_f32.as_slice();
        let mut score = 0.0f64;
        for j in 0..cols {
            let col5 = helix.db_begin_5p + j;
            let col3 = helix.db_begin_3p + cols - 1 - j;
            let code = nt_helix_code(seq[col5], seq[col3]);
            score += prof[code * cols + j] as f64;
        }
        total_score += score;
    }

    total_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;
    use crate::pattern;
    use crate::profile::Background;

    #[test]
    fn test_helix_score_table() {
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let region = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &region, &bg, 0.0002, -20.0);

        // Score the first helix against a short test sequence.
        let helix = &pat.helices[0];
        let seq = b"GGGCGAATAGTGTCAGCGGGAGCACACCAGACTTGCAATCTGGTAGGGAGGGTTCGAGTCCCTCTTTGTCCACCA";
        let table = compute_helix_score_table(seq, helix);

        // Should have max_gaps+1 gap variants.
        assert_eq!(table.len(), helix.max_gaps + 1);
        // Should have scan positions.
        if seq.len() >= helix.max_len {
            assert_eq!(table[0].len(), seq.len() - helix.max_len + 1);
        }
    }
}
