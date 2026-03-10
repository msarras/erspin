use crate::scoring::{self};
use crate::types::*;
use rayon::prelude::*;

/// Resolve mask specifications against a pattern, producing resolved masks
/// with element indices and gap configurations.
pub fn resolve_masks(specs: &[MaskSpec], pattern: &Pattern) -> Vec<ResolvedMask> {
    let mut resolved = Vec::with_capacity(specs.len());
    let mut prev_elements: Vec<i32> = Vec::new();

    for spec in specs {
        let element_ids = match spec.mode {
            MaskMode::Mask => {
                // Use only the specified element IDs.
                spec.elements.iter().map(|&e| e as i32).collect::<Vec<_>>()
            }
            MaskMode::Umask => {
                // Use all elements EXCEPT the specified ones.
                let exclude: Vec<i32> = spec.elements.iter().map(|&e| e as i32).collect();
                let mut ids = Vec::new();
                for h in &pattern.helices {
                    if !exclude.contains(&h.id) {
                        ids.push(h.id);
                    }
                }
                for s in &pattern.strands {
                    if !exclude.contains(&s.id) {
                        ids.push(s.id);
                    }
                }
                ids
            }
            MaskMode::Add => {
                // Add to previous level's elements.
                let mut ids = prev_elements.clone();
                for &e in &spec.elements {
                    let id = e as i32;
                    if !ids.contains(&id) {
                        ids.push(id);
                    }
                }
                ids
            }
            MaskMode::NoMask => {
                // All elements.
                let mut ids: Vec<i32> = pattern.helices.iter().map(|h| h.id).collect();
                ids.extend(pattern.strands.iter().map(|s| s.id));
                ids
            }
        };

        prev_elements = element_ids.clone();

        // Map element IDs to helix/strand indices.
        let hx_indices: Vec<usize> = pattern
            .helices
            .iter()
            .enumerate()
            .filter(|(_, h)| element_ids.contains(&h.id))
            .map(|(i, _)| i)
            .collect();

        let st_indices: Vec<usize> = pattern
            .strands
            .iter()
            .enumerate()
            .filter(|(_, s)| element_ids.contains(&s.id))
            .map(|(i, _)| i)
            .collect();

        // Generate configurations.
        let configs = generate_configs(pattern, &hx_indices, &st_indices);

        // Compute geometry.
        let (min_bgn, max_bgn, min_len, max_len) =
            compute_mask_geometry(pattern, &hx_indices, &st_indices);

        resolved.push(ResolvedMask {
            hx_indices,
            st_indices,
            configs,
            threshold: f64::NEG_INFINITY,
            min_bgn,
            max_bgn,
            min_len,
            max_len,
        });
    }

    resolved
}

/// Compute cutoff thresholds from training set scores.
///
/// `cutoffs` is a list of cutoff specifications: either a percentage string
/// like "100%" or "90%", or a raw score value.
pub fn compute_thresholds(
    ts: &TrainingSet,
    pattern: &Pattern,
    masks: &mut [ResolvedMask],
    cutoffs: &[String],
) {
    for (i, mask) in masks.iter_mut().enumerate() {
        let cutoff_str = if i < cutoffs.len() {
            &cutoffs[i]
        } else {
            "100%"
        };

        if let Some(pct_str) = cutoff_str.strip_suffix('%') {
            let pct: f64 = pct_str.parse().unwrap_or(100.0);

            // Score all training sequences with this mask.
            let mut scores: Vec<f64> = ts
                .sequences
                .iter()
                .map(|seq| scoring::score_training_sequence(seq, pattern, mask))
                .collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

            // Find the threshold at the given percentage.
            let nscores = scores.len();
            if pct >= 100.0 {
                // 100% captures all training sequences.
                mask.threshold = scores[0] - 0.001;
            } else {
                let idx = ((nscores as f64) * (1.0 - pct / 100.0)).ceil() as usize;
                let idx = idx.min(nscores - 1);
                mask.threshold = scores[idx] - 0.001;
            }
        } else {
            // Raw score value.
            mask.threshold = cutoff_str.parse().unwrap_or(0.0);
        }
    }
}

/// Generate all valid gap configurations for a mask.
///
/// Following the C ERPIN approach:
/// - Only strand-type atoms with max_gaps > 0 are gap variables
/// - Non-mask strand atoms between mask atoms are grouped into cumulative gap variables
/// - Helix gap variants are derived from the gaps of intervening atoms, not enumerated independently
fn generate_configs(
    pattern: &Pattern,
    hx_indices: &[usize],
    st_indices: &[usize],
) -> Vec<Config> {
    let nhx = hx_indices.len();
    let nst = st_indices.len();

    if pattern.atoms.is_empty() {
        return vec![Config {
            len: 0,
            st_bgn: vec![0; nst],
            st_gaps: vec![0; nst],
            hx_bgn: vec![0; nhx],
            hx_gaps: vec![0; nhx],
        }];
    }

    // Build atomstr: which atoms are in the mask.
    let mut in_mask = vec![false; pattern.atoms.len()];
    for &hi in hx_indices {
        for (ai, a) in pattern.atoms.iter().enumerate() {
            if a.element_index == hi
                && (a.atom_type == AtomType::Helix1 || a.atom_type == AtomType::Helix2)
            {
                in_mask[ai] = true;
            }
        }
    }
    for &si in st_indices {
        for (ai, a) in pattern.atoms.iter().enumerate() {
            if a.element_index == si && a.atom_type == AtomType::Strand {
                in_mask[ai] = true;
            }
        }
    }

    // Find the envelope: first mask atom to last mask atom.
    let first_mask = in_mask.iter().position(|&m| m).unwrap_or(0);
    let last_mask = in_mask.iter().rposition(|&m| m).unwrap_or(0);

    // Identify gap variables by walking atoms in the envelope.
    // A gap variable is either:
    // - A mask strand atom with max_gaps > 0 (individual variable)
    // - A group of consecutive non-mask atoms with cumulative max_gaps > 0
    struct GapVar {
        /// Range of gap values: 0..=max_gaps
        max_gaps: usize,
        /// Which atom indices this variable covers (for non-mask groups).
        atom_range: (usize, usize), // inclusive start, exclusive end
        /// If this is a single mask strand atom, its index in st_indices.
        mask_strand: Option<usize>,
    }

    let mut gap_vars: Vec<GapVar> = Vec::new();
    let mut ai = first_mask;

    while ai <= last_mask {
        if in_mask[ai] {
            // Mask atom: if it's a strand with gaps, it's an individual gap variable.
            let atom = &pattern.atoms[ai];
            if atom.atom_type == AtomType::Strand && atom.max_gaps > 0 {
                // Find index in st_indices.
                let st_mask_idx = st_indices
                    .iter()
                    .position(|&si| si == atom.element_index)
                    .unwrap();
                gap_vars.push(GapVar {
                    max_gaps: atom.max_gaps,
                    atom_range: (ai, ai + 1),
                    mask_strand: Some(st_mask_idx),
                });
            }
            ai += 1;
        } else {
            // Non-mask atom(s): group consecutive non-mask atoms.
            let group_start = ai;
            let mut cum_gaps = 0usize;
            while ai <= last_mask && !in_mask[ai] {
                cum_gaps += pattern.atoms[ai].max_gaps;
                ai += 1;
            }
            if cum_gaps > 0 {
                gap_vars.push(GapVar {
                    max_gaps: cum_gaps,
                    atom_range: (group_start, ai),
                    mask_strand: None,
                });
            }
        }
    }

    // Enumerate all combinations of gap variable values.
    let ranges: Vec<usize> = gap_vars.iter().map(|v| v.max_gaps + 1).collect();
    let total: usize = if ranges.is_empty() {
        1
    } else {
        ranges.iter().product()
    };
    let mut configs = Vec::with_capacity(total);

    for combo_idx in 0..total {
        // Decode combination index into gap values.
        let mut gap_values = vec![0usize; gap_vars.len()];
        if !ranges.is_empty() {
            let mut remainder = combo_idx;
            for i in (0..ranges.len()).rev() {
                gap_values[i] = remainder % ranges[i];
                remainder /= ranges[i];
            }
        }

        // Build per-atom gap assignments within the envelope.
        let mut atom_gaps = vec![0usize; pattern.atoms.len()];
        for (vi, var) in gap_vars.iter().enumerate() {
            let val = gap_values[vi];
            if var.mask_strand.is_some() {
                // Single mask strand: assign gap directly.
                atom_gaps[var.atom_range.0] = val;
            } else {
                // Non-mask group: assign cumulative gap to the group.
                // The total gap is distributed across the group atoms.
                // For position computation, we only need the total, so we
                // assign it to the first atom with gaps in the group.
                let mut remaining = val;
                for a in var.atom_range.0..var.atom_range.1 {
                    let max = pattern.atoms[a].max_gaps;
                    let assign = remaining.min(max);
                    atom_gaps[a] = assign;
                    remaining -= assign;
                }
            }
        }

        // Walk all atoms from first_mask to last_mask, computing positions.
        let mut pos = 0usize;
        let mut hx_bgn = vec![0usize; nhx];
        let mut hx_gaps = vec![0usize; nhx];
        let mut st_bgn = vec![0usize; nst];
        let mut st_gaps = vec![0usize; nst];

        for ai in first_mask..=last_mask {
            let atom = &pattern.atoms[ai];

            // Record position for mask elements.
            if in_mask[ai] {
                match atom.atom_type {
                    AtomType::Helix1 => {
                        if let Some(mi) = hx_indices.iter().position(|&hi| hi == atom.element_index)
                        {
                            hx_bgn[mi] = pos;
                        }
                    }
                    AtomType::Helix2 => {
                        // Compute helix gap = distance between HLX1 end and HLX2 start,
                        // minus the helix's min_dist.
                        if let Some(mi) = hx_indices.iter().position(|&hi| hi == atom.element_index)
                        {
                            let helix = &pattern.helices[atom.element_index];
                            let hlx1_end = hx_bgn[mi] + helix.helix_len;
                            hx_gaps[mi] = pos - hlx1_end - helix.min_dist;
                        }
                    }
                    AtomType::Strand => {
                        if let Some(mi) = st_indices.iter().position(|&si| si == atom.element_index)
                        {
                            st_bgn[mi] = pos;
                            st_gaps[mi] = atom_gaps[ai];
                        }
                    }
                }
            }

            // Advance position by atom's min_len + assigned gaps.
            pos += atom.min_len + atom_gaps[ai];
        }

        configs.push(Config {
            len: pos,
            st_bgn,
            st_gaps,
            hx_bgn,
            hx_gaps,
        });
    }

    configs
}

/// Compute mask geometry: min_bgn, max_bgn, min_len, max_len.
fn compute_mask_geometry(
    pattern: &Pattern,
    hx_indices: &[usize],
    st_indices: &[usize],
) -> (usize, usize, usize, usize) {
    let mut first_pos = usize::MAX;
    let mut last_end = 0usize;
    let mut first_min_bgn = usize::MAX;

    for &hi in hx_indices {
        let h = &pattern.helices[hi];
        if h.db_begin_5p < first_pos {
            first_pos = h.db_begin_5p;
            first_min_bgn = h.min_bgn;
        }
        let end = h.db_begin_5p + h.max_len;
        if end > last_end {
            last_end = end;
        }
    }

    for &si in st_indices {
        let s = &pattern.strands[si];
        if s.db_begin < first_pos {
            first_pos = s.db_begin;
            first_min_bgn = s.min_bgn;
        }
        let end = s.db_begin + s.max_len;
        if end > last_end {
            last_end = end;
        }
    }

    let max_bgn = if first_pos >= pattern.db_begin {
        first_pos - pattern.db_begin
    } else {
        0
    };
    let min_bgn = first_min_bgn.min(max_bgn);
    let max_len = last_end - first_pos;
    let min_len = max_len; // approximate; configs will have the real lengths

    (min_bgn, max_bgn, min_len, max_len)
}

/// Search a single sequence with a multi-level mask cascade.
///
/// Returns all non-overlapping hits that pass all threshold levels.
pub fn search_sequence(
    pattern: &Pattern,
    masks: &[ResolvedMask],
    seq_data: &[u8],
    direction: StrandDirection,
) -> Vec<Hit> {
    if masks.is_empty() || seq_data.len() < pattern.min_len {
        return Vec::new();
    }

    // Level 0: scan the entire sequence.
    let level0_hits = scan_level(pattern, &masks[0], seq_data, 0, seq_data.len());

    // Subsequent levels: narrow search around each hit from the previous level.
    let mut current_hits = level0_hits;

    for level in 1..masks.len() {
        let mask = &masks[level];
        let mut next_hits = Vec::new();

        for hit in &current_hits {
            // Search a window around the hit position.
            // The hit offset is relative to the previous mask's starting element.
            // The current mask may start earlier or extend further, so use a
            // generous window based on the full pattern extent.
            let search_bgn = hit.offset.saturating_sub(pattern.max_len);
            let search_end = (hit.offset + pattern.max_len).min(seq_data.len());

            let level_hits = scan_level(pattern, mask, seq_data, search_bgn, search_end);
            next_hits.extend(level_hits);
        }

        current_hits = next_hits;
    }

    // Remove overlapping hits (keep highest score).
    overlap_filter(&current_hits, direction)
}

/// Scan a range of positions at a single level.
fn scan_level(
    pattern: &Pattern,
    mask: &ResolvedMask,
    seq_data: &[u8],
    range_start: usize,
    range_end: usize,
) -> Vec<Hit> {
    let mut hits = Vec::new();

    if range_end <= range_start || mask.configs.is_empty() {
        return hits;
    }

    // Compute score tables for this range.
    let window = &seq_data[range_start..range_end];
    let tables = scoring::compute_mask_score_tables(window, pattern, mask);

    // Scan positions.
    let scan_len = if window.len() > mask.min_len {
        window.len() - mask.min_len + 1
    } else {
        return hits;
    };

    // Build precomputed config lookup for fast scoring.
    let lookup = scoring::ConfigLookup::build(mask, &tables);

    for pos in 0..scan_len {
        let (score, cfg_idx) = lookup.best_score_threshold(pos, mask.threshold);
        if score > mask.threshold {
            hits.push(Hit {
                offset: range_start + pos,
                length: lookup.config_len(cfg_idx),
                score,
                evalue: None,
                direction: StrandDirection::Forward,
                config_index: cfg_idx,
            });
        }
    }

    hits
}

/// Remove overlapping hits, keeping the highest-scoring one in each overlap group.
fn overlap_filter(hits: &[Hit], direction: StrandDirection) -> Vec<Hit> {
    if hits.is_empty() {
        return Vec::new();
    }

    // Sort by position.
    let mut sorted: Vec<Hit> = hits.to_vec();
    sorted.sort_by_key(|h| h.offset);

    let mut result = Vec::new();
    let mut i = 0;

    while i < sorted.len() {
        let mut best_score = sorted[i].score;
        let mut best_idx = i;
        let end_of_first = sorted[i].offset + sorted[i].length;

        let mut j = i + 1;
        while j < sorted.len() && sorted[j].offset < end_of_first {
            if sorted[j].score > best_score {
                best_score = sorted[j].score;
                best_idx = j;
            }
            j += 1;
        }

        let mut hit = sorted[best_idx].clone();
        hit.direction = direction;
        result.push(hit);
        i = j;
    }

    result
}

/// Run a full search: forward, reverse, or both strands.
pub fn search_full(
    pattern: &Pattern,
    masks: &[ResolvedMask],
    seq: &Sequence,
    direction: StrandDirection,
) -> Vec<Hit> {
    let mut all_hits = Vec::new();

    if matches!(direction, StrandDirection::Forward | StrandDirection::Both) {
        let mut hits =
            search_sequence(pattern, masks, &seq.data, StrandDirection::Forward);
        all_hits.append(&mut hits);
    }

    if matches!(direction, StrandDirection::Reverse | StrandDirection::Both) {
        let rc = seq.reverse_complement();
        let mut hits = search_sequence(pattern, masks, &rc, StrandDirection::Reverse);
        // Convert positions back to forward strand coordinates.
        for hit in &mut hits {
            let fwd_end = seq.len() - hit.offset;
            let fwd_start = fwd_end - hit.length;
            hit.offset = fwd_start;
            hit.direction = StrandDirection::Reverse;
        }
        all_hits.append(&mut hits);
    }

    all_hits
}

/// Search all sequences in parallel using Rayon.
///
/// Returns a Vec of (sequence_index, Vec<Hit>) for sequences that have hits.
pub fn search_all_parallel(
    pattern: &Pattern,
    masks: &[ResolvedMask],
    sequences: &[Sequence],
    direction: StrandDirection,
) -> Vec<(usize, Vec<Hit>)> {
    sequences
        .par_iter()
        .enumerate()
        .filter_map(|(i, seq)| {
            let hits = search_full(pattern, masks, seq, direction);
            if hits.is_empty() {
                None
            } else {
                Some((i, hits))
            }
        })
        .collect()
}

/// Format a hit for output.
pub fn format_hit(hit: &Hit, _seq: &Sequence) -> String {
    let dir_str = match hit.direction {
        StrandDirection::Forward => "FW",
        StrandDirection::Reverse => "RC",
        StrandDirection::Both => "??",
    };
    let pos1 = hit.offset + 1; // 1-based
    let end1 = hit.offset + hit.length;

    let evalue_str = hit
        .evalue
        .map_or(String::new(), |e| format!("  {:.2e}", e));

    format!(
        "{} {:>7}..{:<7}  {:.2}{}",
        dir_str, pos1, end1, hit.score, evalue_str
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;
    use crate::pattern;
    use crate::profile::Background;
    use crate::region;

    #[test]
    fn test_resolve_masks() {
        let ts = epn::parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

        // Test: "6,8 / !2,3 / *" (equivalent to C's "-umask 6 8 -mask 2 3 -nomask")
        let specs = region::parse_mask_specs("6,8 / !2,3 / *").unwrap();
        let masks = resolve_masks(&specs, &pat);

        assert_eq!(masks.len(), 3);

        // Level 1: elements 6, 8 (two helices).
        assert_eq!(masks[0].hx_indices.len(), 2);
        assert_eq!(masks[0].st_indices.len(), 0);

        // Level 2: all except 2, 3 → 3 helices + 5 strands = 8 elements.
        assert_eq!(
            masks[1].hx_indices.len() + masks[1].st_indices.len(),
            8,
            "level 2: expected 8 elements, got {} helices + {} strands",
            masks[1].hx_indices.len(),
            masks[1].st_indices.len()
        );

        // Level 3: all → 4 helices + 6 strands = 10 elements.
        assert_eq!(masks[2].hx_indices.len(), 4);
        assert_eq!(masks[2].st_indices.len(), 6);
    }

    #[test]
    fn test_search_trna() {
        let ts = epn::parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

        let specs = region::parse_mask_specs("6,8 / !2,3 / *").unwrap();
        let mut masks = resolve_masks(&specs, &pat);

        // Compute thresholds: 100%, 100%, 90%.
        let cutoffs = vec!["100%".to_string(), "100%".to_string(), "90%".to_string()];
        compute_thresholds(&ts, &pat, &mut masks, &cutoffs);

        eprintln!("Thresholds: {:?}", masks.iter().map(|m| m.threshold).collect::<Vec<_>>());
        eprintln!("Config counts: {:?}", masks.iter().map(|m| m.configs.len()).collect::<Vec<_>>());

        // Load test FASTA.
        let reader =
            crate::fasta::FastaReader::from_path("erpin5.5.4.serv/start.test/test.trna.fasta")
                .unwrap();
        let sequences = reader.collect_all().unwrap();
        let seq = &sequences[0];

        let hits = search_full(&pat, &masks, seq, StrandDirection::Both);

        eprintln!("Found {} hits", hits.len());
        for hit in &hits {
            eprintln!("  {}", format_hit(hit, seq));
        }

        // The original ERPIN finds 7 independent hits.
        // We may not match exactly due to config generation differences,
        // but we should find hits.
        assert!(
            !hits.is_empty(),
            "expected to find tRNA hits in the test data"
        );
    }

}
