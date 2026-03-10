use crate::profile::{nt_helix_code, nt_strand_code};
use crate::simd;
use crate::types::*;

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
    let helix_scores: Vec<_> = mask
        .hx_indices
        .iter()
        .map(|&idx| compute_helix_score_table(seq, &pattern.helices[idx]))
        .collect();

    let strand_scores: Vec<_> = mask
        .st_indices
        .iter()
        .map(|&idx| compute_strand_score_table(seq, &pattern.strands[idx]))
        .collect();

    ScoreTables {
        helix_scores,
        strand_scores,
    }
}

/// A precomputed lookup entry for fast config scoring.
/// Each entry holds a slice reference and an offset, avoiding repeated
/// bounds checking and Vec indirection in the hot loop.
pub struct ConfigLookup<'a> {
    /// For each config: a flat array of (score_slice, bgn_offset) pairs.
    /// First strands, then helices. All configs have the same number of entries.
    entries: Vec<Vec<(&'a [f32], usize)>>,
    /// Config lengths.
    config_lens: Vec<usize>,
    /// Suffix max scores: suffix_max[i] = max score achievable from elements i..n.
    /// Used for early termination. Computed per-element as the global max over
    /// all gap variants and positions.
    suffix_max: Vec<f64>,
}

impl<'a> ConfigLookup<'a> {
    /// Build a precomputed config lookup from mask and score tables.
    pub fn build(mask: &ResolvedMask, tables: &'a ScoreTables) -> Self {
        let n_elements = mask.st_indices.len() + mask.hx_indices.len();
        let mut entries = Vec::with_capacity(mask.configs.len());
        let mut config_lens = Vec::with_capacity(mask.configs.len());

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

        for cfg in &mask.configs {
            let mut cfg_entries = Vec::with_capacity(n_elements);
            let mut valid = true;

            for s in 0..mask.st_indices.len() {
                let gap = cfg.st_gaps[s];
                if gap < tables.strand_scores[s].len() {
                    cfg_entries.push((
                        tables.strand_scores[s][gap].as_slice(),
                        cfg.st_bgn[s],
                    ));
                } else {
                    valid = false;
                    break;
                }
            }

            if valid {
                for h in 0..mask.hx_indices.len() {
                    let gap = cfg.hx_gaps[h];
                    if gap < tables.helix_scores[h].len() {
                        cfg_entries.push((
                            tables.helix_scores[h][gap].as_slice(),
                            cfg.hx_bgn[h],
                        ));
                    } else {
                        valid = false;
                        break;
                    }
                }
            }

            if !valid {
                cfg_entries.clear();
            }
            entries.push(cfg_entries);
            config_lens.push(cfg.len);
        }

        Self { entries, config_lens, suffix_max }
    }

    /// Find the best-scoring configuration at a given position.
    #[inline]
    pub fn best_score(&self, tab_index: usize) -> (f64, usize) {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_idx = 0;

        for (cfg_idx, cfg_entries) in self.entries.iter().enumerate() {
            if cfg_entries.is_empty() {
                continue;
            }
            let mut score = 0.0f64;
            let mut valid = true;

            for (ei, &(slice, bgn)) in cfg_entries.iter().enumerate() {
                let pos = tab_index + bgn;
                if pos < slice.len() {
                    score += unsafe { *slice.get_unchecked(pos) } as f64;
                    // Early termination: if partial score + max possible remaining
                    // can't beat the current best, skip this config.
                    if score + self.suffix_max[ei + 1] <= best_score {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }

            if valid && score > best_score {
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

        for (cfg_idx, cfg_entries) in self.entries.iter().enumerate() {
            if cfg_entries.is_empty() {
                continue;
            }
            let mut score = 0.0f64;
            let mut valid = true;

            for (ei, &(slice, bgn)) in cfg_entries.iter().enumerate() {
                let pos = tab_index + bgn;
                if pos < slice.len() {
                    score += unsafe { *slice.get_unchecked(pos) } as f64;
                    // Early termination: can't reach threshold OR current best.
                    let min_target = if best_score > threshold { best_score } else { threshold };
                    if score + self.suffix_max[ei + 1] <= min_target {
                        valid = false;
                        break;
                    }
                } else {
                    valid = false;
                    break;
                }
            }

            if valid && score > best_score {
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

/// Score a single training set sequence against a mask.
/// Used for threshold computation.
pub fn score_training_sequence(
    seq: &[u8],
    pattern: &Pattern,
    mask: &ResolvedMask,
) -> f64 {
    // The training sequence is aligned, so we score at position 0 with specific column mapping.
    // For each element, extract the non-gap bases and score them.
    let mut total_score = 0.0;

    for (_s, &st_idx) in mask.st_indices.iter().enumerate() {
        let strand = &pattern.strands[st_idx];
        let mut score = 0.0;
        for j in 0..strand.max_len {
            let col = strand.db_begin + j;
            let code = nt_strand_code(seq[col]);
            score += strand.profile[code][j];
        }
        total_score += score;
    }

    for (_h, &hx_idx) in mask.hx_indices.iter().enumerate() {
        let helix = &pattern.helices[hx_idx];
        let mut score = 0.0;
        for j in 0..helix.helix_len {
            let col5 = helix.db_begin_5p + j;
            let col3 = helix.db_begin_3p + helix.helix_len - 1 - j;
            let code = nt_helix_code(seq[col5], seq[col3]);
            score += helix.profile[code][j];
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
        let ts = epn::parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
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
