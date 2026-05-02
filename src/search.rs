use std::cell::RefCell;
use std::sync::Arc;

use crate::scoring;
use crate::types::*;
use rayon::prelude::*;

thread_local! {
    /// Per-thread reusable reverse-complement buffer. `search_full` populates
    /// this for each `RC` pass instead of allocating a fresh Vec.
    static RC_BUFFER: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };

    /// Per-thread scratch score tables. `scan_level` fills this in-place each
    /// time it runs so the per-element / per-gap-variant `Vec<f32>` buffers
    /// stay hot across thousands of chunks instead of being reallocated.
    static SCRATCH_TABLES: RefCell<scoring::ScoreTables> = RefCell::new(scoring::ScoreTables {
        helix_scores: Vec::new(),
        strand_scores: Vec::new(),
    });
}

// ═══════════════════════════════════════════════════════════════════════════════
// Mask Resolution
// ═══════════════════════════════════════════════════════════════════════════════

/// Resolve a single mask spec into element IDs, given the pattern and the
/// previous level's elements (for `Add` mode).
fn resolve_element_ids(
    spec: &MaskSpec,
    pattern: &Pattern,
    prev_elements: &[i32],
) -> Vec<i32> {
    match spec.mode {
        MaskMode::Mask => spec.elements.iter().map(|&e| e as i32).collect(),
        MaskMode::Umask => {
            let exclude: Vec<i32> = spec.elements.iter().map(|&e| e as i32).collect();
            pattern
                .helices
                .iter()
                .map(|h| h.id)
                .chain(pattern.strands.iter().map(|s| s.id))
                .filter(|id| !exclude.contains(id))
                .collect()
        }
        MaskMode::Add => {
            let mut ids = prev_elements.to_vec();
            for &e in &spec.elements {
                let id = e as i32;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            ids
        }
        MaskMode::NoMask => pattern
            .helices
            .iter()
            .map(|h| h.id)
            .chain(pattern.strands.iter().map(|s| s.id))
            .collect(),
    }
}

/// Resolve mask specifications against a pattern, producing resolved masks
/// with element indices and gap configurations.
pub fn resolve_masks(specs: &[MaskSpec], pattern: &Pattern) -> Vec<ResolvedMask> {
    let mut resolved = Vec::with_capacity(specs.len());
    let mut prev_elements: Vec<i32> = Vec::new();

    for (level, spec) in specs.iter().enumerate() {
        let element_ids = resolve_element_ids(spec, pattern, &prev_elements);
        prev_elements = element_ids.clone();

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

        // Build inverse maps for O(1) element-index lookups in the hot path
        // (build_config_from_gaps + collect_gap_variables previously did
        // O(n) iter().position() per atom — those are now O(1)).
        let mut hx_inv: Vec<Option<usize>> = vec![None; pattern.helices.len()];
        for (mi, &hi) in hx_indices.iter().enumerate() {
            hx_inv[hi] = Some(mi);
        }
        let mut st_inv: Vec<Option<usize>> = vec![None; pattern.strands.len()];
        for (mi, &si) in st_indices.iter().enumerate() {
            st_inv[si] = Some(mi);
        }

        let in_mask: Vec<bool> = pattern
            .atoms
            .iter()
            .map(|atom| match atom.atom_type {
                AtomType::Helix1 | AtomType::Helix2 => hx_inv[atom.element_index].is_some(),
                AtomType::Strand => st_inv[atom.element_index].is_some(),
            })
            .collect();
        let first_atom = in_mask.iter().position(|&m| m).unwrap_or(0);
        let last_atom = in_mask.iter().rposition(|&m| m).unwrap_or(0);

        let gap_vars = collect_gap_variables(pattern, &in_mask, first_atom, last_atom);

        // Precompute configs if the count is manageable. Otherwise, leave
        // configs empty; the search will use DMP (per-hit constrained config
        // generation) for that level.
        let total: usize = gap_vars
            .iter()
            .map(|v| v.max_gaps + 1)
            .product::<usize>()
            .max(1);
        let configs = if level == 0 || total <= 100_000 {
            generate_configs(
                pattern,
                &hx_indices,
                &st_indices,
                &hx_inv,
                &st_inv,
                &in_mask,
                first_atom,
                last_atom,
                &gap_vars,
            )
        } else {
            Vec::new() // DMP will handle this level
        };
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
            in_mask,
            first_atom,
            last_atom,
            hx_inv,
            st_inv,
            gap_vars,
        });
    }

    resolved
}

/// Compute cutoff thresholds from training set scores.
///
/// Each cutoff is either a percentage string like `"90%"` (captures that
/// fraction of training sequences) or a raw score value.
pub fn compute_thresholds(
    ts: &TrainingSet,
    pattern: &Pattern,
    masks: &mut [ResolvedMask],
    cutoffs: &[String],
) {
    // Masks are independent → process in parallel. Within each mask, training
    // sequences are independent too, so the inner score computation also runs
    // in parallel via `par_iter`. Rayon's work-stealing handles the nested
    // parallelism without oversubscription.
    masks.par_iter_mut().enumerate().for_each(|(i, mask)| {
        let cutoff_str = cutoffs.get(i).map_or("100%", String::as_str);

        if let Some(pct_str) = cutoff_str.strip_suffix('%') {
            let pct: f64 = pct_str.parse().unwrap_or(100.0);

            let mut scores: Vec<f64> = ts
                .sequences
                .par_iter()
                .map(|seq| scoring::score_training_sequence(seq, pattern, mask))
                .collect();
            scores.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            // Mirror C ERPIN's `ConvertRatio` (libsrc/tscores.c:262): walk
            // sorted-ascending scores, take the first index `j` where
            // round(100 * (N - j) / N) <= percent, and return ts_scores[j-1]
            // (or ts_scores[0] if j == 0). The original Rust used a simpler
            // `ceil((1 - pct/100) * N)` formula which is off-by-one vs C and
            // produced a slightly stricter threshold.
            let n_scores = scores.len();
            let percent = pct as i32;
            let mut idx = n_scores - 1; // initial: highest (= "% minimal")
            for j in 0..n_scores {
                let ratio = ((100.0 * (n_scores - j) as f64 / n_scores as f64).round()) as i32;
                if ratio <= percent {
                    idx = if j > 0 { j - 1 } else { 0 };
                    break;
                }
            }
            mask.threshold = scores[idx] - 1e-3;
        } else {
            mask.threshold = cutoff_str.parse().unwrap_or(0.0);
        }
    });
}

/// Generate all valid gap configurations for a mask.
///
/// Following the C ERPIN approach:
/// - Only strand-type atoms with max_gaps > 0 are gap variables
/// - Non-mask strand atoms between mask atoms are grouped into cumulative
///   gap variables
/// - Helix gap variants are derived from the gaps of intervening atoms,
///   not enumerated independently
#[allow(clippy::too_many_arguments)]
fn generate_configs(
    pattern: &Pattern,
    hx_indices: &[usize],
    st_indices: &[usize],
    hx_inv: &[Option<usize>],
    st_inv: &[Option<usize>],
    in_mask: &[bool],
    first_mask: usize,
    last_mask: usize,
    gap_vars: &[GapVar],
) -> Vec<Config> {
    let nhx = hx_indices.len();
    let nst = st_indices.len();

    if pattern.atoms.is_empty() || !in_mask.iter().any(|&m| m) {
        return vec![Config {
            len: 0,
            st_bgn: vec![0; nst],
            st_gaps: vec![0; nst],
            hx_bgn: vec![0; nhx],
            hx_gaps: vec![0; nhx],
            atom_gaps: Arc::from(vec![0usize; pattern.atoms.len()].into_boxed_slice()),
        }];
    }

    // Enumerate all combinations of gap variable values.
    let ranges: Vec<usize> = gap_vars.iter().map(|v| v.max_gaps + 1).collect();
    let total: usize = ranges.iter().copied().product::<usize>().max(1);
    let mut configs = Vec::with_capacity(total);

    let natoms = pattern.atoms.len();
    let mut atom_gaps = vec![0usize; natoms];
    let mut gap_values = vec![0usize; gap_vars.len()];

    for combo_idx in 0..total {
        decode_combination_into(combo_idx, &ranges, &mut gap_values);

        atom_gaps.iter_mut().for_each(|g| *g = 0);
        distribute_gaps_into(pattern, gap_vars, &gap_values, &mut atom_gaps);

        let config = build_config_from_gaps(
            pattern,
            nhx,
            nst,
            hx_inv,
            st_inv,
            in_mask,
            &atom_gaps,
            first_mask,
            last_mask,
        );
        configs.push(config);
    }

    configs
}

/// Generate configs for a mask level, constraining gap variables that were
/// within the previous mask's envelope to their winning values (DMP).
///
/// `prev_atom_gaps` contains the per-atom gap assignments from the winning
/// config of the previous level. Gap variables whose atom range falls within
/// `prev_first..=prev_last` are fixed; others are enumerated freely.
fn generate_configs_constrained(
    pattern: &Pattern,
    mask: &ResolvedMask,
    prev_atom_gaps: &[usize],
    prev_first: usize,
    prev_last: usize,
) -> Vec<Config> {
    let nhx = mask.hx_indices.len();
    let nst = mask.st_indices.len();
    let in_mask = &mask.in_mask;
    let first_mask = mask.first_atom;
    let last_mask = mask.last_atom;
    let gap_vars = &mask.gap_vars;

    if !in_mask.iter().any(|&m| m) {
        return vec![Config {
            len: 0,
            st_bgn: vec![0; nst],
            st_gaps: vec![0; nst],
            hx_bgn: vec![0; nhx],
            hx_gaps: vec![0; nhx],
            atom_gaps: Arc::from(vec![0usize; pattern.atoms.len()].into_boxed_slice()),
        }];
    }

    // Split variables into fixed (within previous envelope) and free.
    let mut free_indices = Vec::with_capacity(gap_vars.len());
    let mut fixed_values = vec![0usize; gap_vars.len()];

    for (vi, var) in gap_vars.iter().enumerate() {
        if var.atom_range.0 >= prev_first && var.atom_range.1 <= prev_last + 1 {
            // Variable falls within previous envelope: fix to the sum of
            // previous atom gaps over its range.
            let sum: usize = (var.atom_range.0..var.atom_range.1)
                .map(|a| prev_atom_gaps[a])
                .sum();
            fixed_values[vi] = sum;
        } else {
            free_indices.push(vi);
        }
    }

    let free_ranges: Vec<usize> =
        free_indices.iter().map(|&i| gap_vars[i].max_gaps + 1).collect();
    let total: usize = free_ranges.iter().copied().product::<usize>().max(1);
    let mut configs = Vec::with_capacity(total);

    let natoms = pattern.atoms.len();
    let mut gap_values = fixed_values.clone();
    let mut free_values = vec![0usize; free_indices.len()];
    let mut atom_gaps = vec![0usize; natoms];

    for combo_idx in 0..total {
        decode_combination_into(combo_idx, &free_ranges, &mut free_values);

        // Reset gap_values to fixed_values (only the free positions change).
        for (fi, &vi) in free_indices.iter().enumerate() {
            gap_values[vi] = free_values[fi];
        }

        atom_gaps.iter_mut().for_each(|g| *g = 0);
        distribute_gaps_into(pattern, gap_vars, &gap_values, &mut atom_gaps);

        let config = build_config_from_gaps(
            pattern,
            nhx,
            nst,
            &mask.hx_inv,
            &mask.st_inv,
            in_mask,
            &atom_gaps,
            first_mask,
            last_mask,
        );
        configs.push(config);
    }

    configs
}

/// Walk the mask envelope and collect gap variables.
fn collect_gap_variables(
    pattern: &Pattern,
    in_mask: &[bool],
    first: usize,
    last: usize,
) -> Vec<GapVar> {
    let mut vars = Vec::new();
    if !in_mask.iter().any(|&m| m) {
        return vars;
    }
    let mut ai = first;

    while ai <= last {
        if in_mask[ai] {
            let atom = &pattern.atoms[ai];
            if atom.atom_type == AtomType::Strand && atom.max_gaps > 0 {
                vars.push(GapVar {
                    max_gaps: atom.max_gaps,
                    atom_range: (ai, ai + 1),
                    is_mask_strand: true,
                });
            }
            ai += 1;
        } else {
            let group_start = ai;
            let mut cum_gaps = 0usize;
            while ai <= last && !in_mask[ai] {
                cum_gaps += pattern.atoms[ai].max_gaps;
                ai += 1;
            }
            if cum_gaps > 0 {
                vars.push(GapVar {
                    max_gaps: cum_gaps,
                    atom_range: (group_start, ai),
                    is_mask_strand: false,
                });
            }
        }
    }

    vars
}

/// Decode a linear combination index into a caller-provided slot, avoiding
/// the per-call Vec allocation.
fn decode_combination_into(mut combo_idx: usize, ranges: &[usize], values: &mut [usize]) {
    debug_assert_eq!(values.len(), ranges.len());
    for i in (0..ranges.len()).rev() {
        values[i] = combo_idx % ranges[i];
        combo_idx /= ranges[i];
    }
}

/// Distribute gap variable values into a caller-provided `atom_gaps` buffer.
/// The buffer must already be zeroed for atoms touched by no variable.
fn distribute_gaps_into(
    pattern: &Pattern,
    gap_vars: &[GapVar],
    gap_values: &[usize],
    atom_gaps: &mut [usize],
) {
    for (vi, var) in gap_vars.iter().enumerate() {
        let val = gap_values[vi];
        if var.is_mask_strand {
            atom_gaps[var.atom_range.0] = val;
        } else {
            let mut remaining = val;
            for a in var.atom_range.0..var.atom_range.1 {
                let assign = remaining.min(pattern.atoms[a].max_gaps);
                atom_gaps[a] = assign;
                remaining -= assign;
            }
        }
    }
}

/// Walk the mask envelope and build a Config from per-atom gap assignments.
/// Uses precomputed inverse-index slices (`hx_inv`, `st_inv`) so element
/// position lookups are O(1) instead of O(mask elements).
#[allow(clippy::too_many_arguments)]
fn build_config_from_gaps(
    pattern: &Pattern,
    nhx: usize,
    nst: usize,
    hx_inv: &[Option<usize>],
    st_inv: &[Option<usize>],
    in_mask: &[bool],
    atom_gaps: &[usize],
    first: usize,
    last: usize,
) -> Config {
    let mut pos = 0usize;
    let mut hx_bgn = vec![0usize; nhx];
    let mut hx_gaps = vec![0usize; nhx];
    let mut st_bgn = vec![0usize; nst];
    let mut st_gaps = vec![0usize; nst];

    for ai in first..=last {
        let atom = &pattern.atoms[ai];

        if in_mask[ai] {
            match atom.atom_type {
                AtomType::Helix1 => {
                    if let Some(mi) = hx_inv[atom.element_index] {
                        hx_bgn[mi] = pos;
                    }
                }
                AtomType::Helix2 => {
                    if let Some(mi) = hx_inv[atom.element_index] {
                        let helix = &pattern.helices[atom.element_index];
                        let hlx1_end = hx_bgn[mi] + helix.helix_len;
                        hx_gaps[mi] = pos - hlx1_end - helix.min_dist;
                    }
                }
                AtomType::Strand => {
                    if let Some(mi) = st_inv[atom.element_index] {
                        st_bgn[mi] = pos;
                        st_gaps[mi] = atom_gaps[ai];
                    }
                }
            }
        }

        pos += atom.min_len + atom_gaps[ai];
    }

    Config {
        len: pos,
        st_bgn,
        st_gaps,
        hx_bgn,
        hx_gaps,
        atom_gaps: Arc::from(atom_gaps.to_vec().into_boxed_slice()),
    }
}

/// Compute mask geometry: min_bgn, max_bgn, min_len, max_len.
fn compute_mask_geometry(
    pattern: &Pattern,
    hx_indices: &[usize],
    st_indices: &[usize],
) -> (usize, usize, usize, usize) {
    // Combine helix and strand elements into (db_begin, min_bgn, end) triples.
    let helix_spans = hx_indices.iter().map(|&i| {
        let h = &pattern.helices[i];
        (h.db_begin_5p, h.min_bgn, h.db_begin_5p + h.max_len)
    });
    let strand_spans = st_indices.iter().map(|&i| {
        let s = &pattern.strands[i];
        (s.db_begin, s.min_bgn, s.db_begin + s.max_len)
    });

    let first = helix_spans
        .clone()
        .chain(strand_spans.clone())
        .min_by_key(|&(db_begin, _, _)| db_begin);

    let Some((first_pos, first_min_bgn, _)) = first else {
        return (0, 0, 0, 0);
    };

    let last_end = helix_spans
        .chain(strand_spans)
        .map(|(_, _, end)| end)
        .max()
        .unwrap_or(0);

    let max_bgn = first_pos.saturating_sub(pattern.db_begin);
    let min_bgn = first_min_bgn.min(max_bgn);
    let max_len = last_end - first_pos;

    (min_bgn, max_bgn, max_len, max_len)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cascade Search
// ═══════════════════════════════════════════════════════════════════════════════

/// Search a single sequence with a multi-level mask cascade.
///
/// Level 0 uses precomputed configs (SMP). Levels 1+ use dynamic mask
/// processing (DMP): for each hit from the previous level, constrained
/// configs are generated based on the winning config's gap assignments.
pub fn search_sequence(
    pattern: &Pattern,
    masks: &[ResolvedMask],
    seq_data: &[u8],
    direction: StrandDirection,
) -> Vec<Hit> {
    if masks.is_empty() || seq_data.len() < pattern.min_len {
        return Vec::new();
    }

    // Level 0: scan the entire sequence with precomputed configs.
    let mut current_hits = scan_level(pattern, &masks[0], seq_data, 0, seq_data.len());

    // Subsequent levels: SMP if configs were precomputed, DMP otherwise.
    for level in 1..masks.len() {
        let next_mask = &masks[level];

        if !next_mask.configs.is_empty() {
            // SMP: precomputed configs — scan around each hit.
            let mut next_hits = Vec::new();
            for hit in &current_hits {
                let search_bgn = hit.offset.saturating_sub(pattern.max_len);
                let search_end = (hit.offset + pattern.max_len).min(seq_data.len());
                next_hits.extend(scan_level(
                    pattern, next_mask, seq_data, search_bgn, search_end,
                ));
            }
            current_hits = next_hits;
        } else {
            // DMP: generate constrained configs per-hit and score directly.
            let prev_mask = &masks[level - 1];
            let prev_first = prev_mask.first_atom;
            let prev_last = prev_mask.last_atom;

            // DMP hits are independent — process in parallel.
            let next_hits: Vec<Hit> = current_hits
                .par_iter()
                .filter_map(|hit| {
                    let constrained_configs = generate_configs_constrained(
                        pattern,
                        next_mask,
                        &hit.atom_gaps,
                        prev_first,
                        prev_last,
                    );

                    if constrained_configs.is_empty() {
                        return None;
                    }

                    let (score, cfg_idx) = scoring::score_configs_direct_with(
                        seq_data,
                        pattern,
                        next_mask,
                        &constrained_configs,
                        hit.offset,
                        next_mask.threshold,
                    );
                    (score > next_mask.threshold).then(|| Hit {
                        offset: hit.offset,
                        length: constrained_configs[cfg_idx].len,
                        score,
                        evalue: None,
                        direction: StrandDirection::Forward,
                        config_index: cfg_idx,
                        atom_gaps: Arc::clone(&constrained_configs[cfg_idx].atom_gaps),
                    })
                })
                .collect();
            current_hits = next_hits;
        }
    }

    overlap_filter(current_hits, direction)
}

/// Scan a range of positions at a single mask level.
fn scan_level(
    pattern: &Pattern,
    mask: &ResolvedMask,
    seq_data: &[u8],
    range_start: usize,
    range_end: usize,
) -> Vec<Hit> {
    if range_end <= range_start || mask.configs.is_empty() {
        return Vec::new();
    }

    let window = &seq_data[range_start..range_end];
    let Some(scan_len) = window.len().checked_sub(mask.min_len) else {
        return Vec::new();
    };
    let scan_len = scan_len + 1;

    SCRATCH_TABLES.with(|cell| {
        let mut tables = cell.borrow_mut();
        scoring::compute_mask_score_tables_into(window, pattern, mask, &mut tables);
        let lookup = scoring::ConfigLookup::build(mask, &tables);

        (0..scan_len)
            .filter_map(|pos| {
                let (score, cfg_idx) = lookup.best_score_threshold(pos, mask.threshold);
                (score > mask.threshold).then(|| Hit {
                    offset: range_start + pos,
                    length: lookup.config_len(cfg_idx),
                    score,
                    evalue: None,
                    direction: StrandDirection::Forward,
                    config_index: cfg_idx,
                    atom_gaps: Arc::clone(&mask.configs[cfg_idx].atom_gaps),
                })
            })
            .collect()
    })
}


// ═══════════════════════════════════════════════════════════════════════════════
// Hit Processing
// ═══════════════════════════════════════════════════════════════════════════════

/// Remove overlapping hits, keeping the highest-scoring one in each group.
///
/// Takes ownership of the input Vec so the upfront `to_vec()` is gone — for
/// a 100k-hit chunk this drops one big clone pass. The per-group `clone` of
/// the chosen Hit remains, but with `atom_gaps` shared via `Arc<[usize]>`
/// each clone is now a small memcpy + a refcount bump.
fn overlap_filter(mut hits: Vec<Hit>, direction: StrandDirection) -> Vec<Hit> {
    if hits.is_empty() {
        return hits;
    }

    hits.sort_unstable_by_key(|h| h.offset);

    let mut result = Vec::with_capacity(hits.len());
    let mut i = 0;

    while i < hits.len() {
        let mut best_idx = i;
        let end_of_first = hits[i].offset + hits[i].length;

        let mut j = i + 1;
        while j < hits.len() && hits[j].offset < end_of_first {
            if hits[j].score > hits[best_idx].score {
                best_idx = j;
            }
            j += 1;
        }

        let mut hit = hits[best_idx].clone();
        hit.direction = direction;
        result.push(hit);
        i = j;
    }

    result
}

// ═══════════════════════════════════════════════════════════════════════════════
// Parallel Execution
// ═══════════════════════════════════════════════════════════════════════════════

/// A chunk of a sequence for parallel search.
struct Chunk {
    /// Byte range in the original sequence (includes padding).
    data_start: usize,
    data_end: usize,
    /// Owned region — only hits starting here are kept.
    owned_start: usize,
    owned_end: usize,
}

/// Plan how to split a sequence into overlapping chunks for parallel search.
///
/// Each chunk gets `pad` bytes of padding on both sides so the multi-level
/// cascade has full context for hits in the owned region. Returns `None` if
/// the sequence is too small or single-threaded.
fn plan_chunks(seq_len: usize, pad: usize) -> Option<Vec<Chunk>> {
    let min_owned = pad * 5;

    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    if num_cpus <= 1 || seq_len < min_owned * 2 {
        return None;
    }

    // Target 2x CPUs for load-balancing via work-stealing.
    let target_chunks = num_cpus * 2;
    let owned_size = (seq_len / target_chunks).max(min_owned);

    if owned_size >= seq_len {
        return None;
    }

    let mut chunks = Vec::new();
    let mut owned_start = 0;
    loop {
        let owned_end = (owned_start + owned_size).min(seq_len);
        let data_start = owned_start.saturating_sub(pad);
        let data_end = (owned_end + pad).min(seq_len);
        chunks.push(Chunk {
            data_start,
            data_end,
            owned_start,
            owned_end,
        });
        if owned_end >= seq_len {
            break;
        }
        owned_start = owned_end;
    }

    Some(chunks)
}

/// Search a sequence, using parallel chunking when the sequence is large
/// enough to benefit. Produces results identical to sequential search.
fn search_sequence_chunked(
    pattern: &Pattern,
    masks: &[ResolvedMask],
    seq_data: &[u8],
    direction: StrandDirection,
) -> Vec<Hit> {
    let Some(chunks) = plan_chunks(seq_data.len(), pattern.max_len) else {
        return search_sequence(pattern, masks, seq_data, direction);
    };

    let chunk_hits: Vec<Vec<Hit>> = chunks
        .par_iter()
        .map(|chunk| {
            let data = &seq_data[chunk.data_start..chunk.data_end];
            search_sequence(pattern, masks, data, direction)
                .into_iter()
                .filter_map(|mut hit| {
                    hit.offset += chunk.data_start;
                    (hit.offset >= chunk.owned_start && hit.offset < chunk.owned_end)
                        .then_some(hit)
                })
                .collect()
        })
        .collect();

    let merged: Vec<Hit> = chunk_hits.into_iter().flatten().collect();
    overlap_filter(merged, direction)
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
        all_hits.extend(search_sequence_chunked(
            pattern,
            masks,
            &seq.data,
            StrandDirection::Forward,
        ));
    }

    if matches!(direction, StrandDirection::Reverse | StrandDirection::Both) {
        let seq_len = seq.len();
        // Reuse a per-thread reverse-complement buffer instead of allocating
        // a fresh Vec<u8> per sequence. On a 1 GB FASTA scanned across N
        // threads this trades millions of small allocations for N grows.
        let mut hits = RC_BUFFER.with(|cell| {
            let mut rc = cell.borrow_mut();
            seq.reverse_complement_into(&mut rc);
            search_sequence_chunked(pattern, masks, &rc, StrandDirection::Reverse)
        });
        for hit in &mut hits {
            hit.offset = seq_len - hit.offset - hit.length;
        }
        all_hits.extend(hits);
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
            (!hits.is_empty()).then_some((i, hits))
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;
    use crate::pattern;
    use crate::profile::Background;
    use crate::region;

    fn setup_trna() -> (Pattern, Vec<ResolvedMask>) {
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

        let specs = region::parse_mask_specs("6,8 / !2,3 / *").unwrap();
        let mut masks = resolve_masks(&specs, &pat);
        let cutoffs = vec!["100%".to_string(), "100%".to_string(), "90%".to_string()];
        compute_thresholds(&ts, &pat, &mut masks, &cutoffs);

        (pat, masks)
    }

    fn format_hit(hit: &Hit) -> String {
        let dir = match hit.direction {
            StrandDirection::Forward => "FW",
            StrandDirection::Reverse => "RC",
            StrandDirection::Both => "??",
        };
        let evalue_str = hit
            .evalue
            .map_or(String::new(), |e| format!("  {:.2e}", e));
        format!(
            "{} {:>7}..{:<7}  {:.2}{}",
            dir,
            hit.offset + 1,
            hit.offset + hit.length,
            hit.score,
            evalue_str
        )
    }

    #[test]
    fn test_resolve_masks() {
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

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
        let (pat, masks) = setup_trna();

        let reader =
            crate::fasta::FastaReader::from_path("tests/data/test.trna.fasta")
                .unwrap();
        let sequences = reader.collect_all().unwrap();
        let seq = &sequences[0];

        let hits = search_full(&pat, &masks, seq, StrandDirection::Both);

        eprintln!("Found {} hits", hits.len());
        for hit in &hits {
            eprintln!("  {}", format_hit(hit));
        }

        assert!(
            !hits.is_empty(),
            "expected to find tRNA hits in the test data"
        );
    }

    #[test]
    fn test_chunked_vs_unchunked_identical() {
        let (pat, masks) = setup_trna();

        let reader =
            crate::fasta::FastaReader::from_path("tests/data/test.trna.fasta")
                .unwrap();
        let sequences = reader.collect_all().unwrap();
        let base_data = &sequences[0].data;

        // Repeat ~10x to make it large enough to trigger chunking.
        let big_data: Vec<u8> = base_data.repeat(10);

        let unchunked = search_sequence(&pat, &masks, &big_data, StrandDirection::Forward);
        let chunked = search_sequence_chunked(&pat, &masks, &big_data, StrandDirection::Forward);

        assert_eq!(
            unchunked.len(),
            chunked.len(),
            "chunked found {} hits vs unchunked {} hits",
            chunked.len(),
            unchunked.len()
        );

        for (i, (u, c)) in unchunked.iter().zip(chunked.iter()).enumerate() {
            assert_eq!(
                u.offset, c.offset,
                "hit {} offset mismatch: unchunked={} chunked={}",
                i, u.offset, c.offset
            );
            assert!(
                (u.score - c.score).abs() < 1e-6,
                "hit {} score mismatch: unchunked={} chunked={}",
                i, u.score, c.score
            );
        }
    }
}
