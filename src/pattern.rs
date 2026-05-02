use crate::profile::{self, Background, DINUC_LUT};
use crate::types::*;

/// Maximum partial sum of per-column maxima over valid lengths
/// `[min_cols, max_cols]`. Serves as an upper bound on the score this element
/// can ever contribute across all configs, used by DMP early termination.
///
/// For elements with a fixed scoring length (`min_cols == max_cols`), this is
/// just the full sum. For length-variable strands (`min_cols < max_cols`), the
/// trailing columns can have NEGATIVE per-column maxima (sparsely-observed
/// alignment positions), so the full sum can be LESS than the sum over the
/// first `min_cols` columns. Returning the maximum partial sum keeps the
/// bound a true upper bound — the previous "always sum to max_cols" version
/// could underestimate, causing valid configs to be wrongly pruned in
/// `score_configs_direct`.
fn column_max_sum(profile: &[Vec<f64>], min_cols: usize, max_cols: usize) -> f64 {
    let mut running = 0.0f64;
    let mut best = f64::NEG_INFINITY;
    for j in 0..max_cols {
        let mut col_max = f64::NEG_INFINITY;
        for row in profile.iter() {
            if j >= row.len() {
                continue;
            }
            let v = row[j];
            if v > col_max {
                col_max = v;
            }
        }
        if col_max.is_finite() {
            running += col_max;
        }
        if j + 1 >= min_cols && running > best {
            best = running;
        }
    }
    if best.is_finite() { best } else { 0.0 }
}

/// Flatten a `[code][position]` log-odds profile to a flat `f32` buffer
/// laid out as `flat[code * cols + j]`. Skips trailing rows whose lengths
/// don't match `cols`, so this works for both the 6×L strand and 24×L helix
/// shapes. The result is what the SIMD scoring routines consume.
fn flatten_profile_f32(profile: &[Vec<f64>], cols: usize) -> Vec<f32> {
    let codes = profile.len();
    let mut flat = vec![0.0f32; codes * cols];
    for (code, row) in profile.iter().enumerate() {
        debug_assert_eq!(
            row.len(),
            cols,
            "profile row {} has length {}, expected {}",
            code,
            row.len(),
            cols
        );
        for (j, &v) in row.iter().enumerate() {
            flat[code * cols + j] = v as f32;
        }
    }
    flat
}

/// Build a column-major transposed profile with `stride` elements per column
/// (≥ profile.len(); extra slots zero-padded). Layout: `flat[j * stride + code]`.
/// This is what the SIMD score-table routines used to stage per call into
/// `col_vals`; caching it on the Strand/Helix saves the per-call O(cols *
/// stride) copy and lets the gapped-strand DP do cache-friendly per-lane
/// gathers (each column's row of `stride` f32 fits in a single cache line).
fn flatten_profile_f32_t(profile: &[Vec<f64>], cols: usize, stride: usize) -> Vec<f32> {
    debug_assert!(profile.len() <= stride);
    let mut flat = vec![0.0f32; cols * stride];
    for (code, row) in profile.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            if j < cols {
                flat[j * stride + code] = v as f32;
            }
        }
    }
    flat
}

/// Build a per-column dinucleotide pair score table for a helix, stride 64
/// (8 c5 classes × 8 c3 classes, with the upper rows/cols zero-padded).
/// `pair[j * 64 + (c5 * 8 + c3)] = profile[DINUC_LUT[c5][c3]][j]`. Folds the
/// two-step lookup `(c5, c3) → DINUC_LUT → profile_f32_t` into a single read
/// in the SIMD helix score-table inner loop. Building this once amortizes
/// the dinuc indirection over thousands of scan windows.
fn build_helix_pair_table(profile: &[Vec<f64>], helix_len: usize) -> Vec<f32> {
    let mut pair = vec![0.0f32; helix_len * 64];
    for j in 0..helix_len {
        for c5 in 0..7 {
            for c3 in 0..7 {
                let code = DINUC_LUT[c5][c3] as usize;
                let v = if code < profile.len() && j < profile[code].len() {
                    profile[code][j] as f32
                } else {
                    0.0
                };
                pair[j * 64 + (c5 * 8 + c3)] = v;
            }
        }
    }
    pair
}

/// Build a Pattern from a training set and region specification.
///
/// The pattern contains all structural elements (helices and strands) within
/// the specified region, along with their scoring profiles.
pub fn build_pattern(
    ts: &TrainingSet,
    region: &Region,
    bg: &Background,
    pseudo_count_weight: f64,
    log_zero: f64,
) -> Pattern {
    // 1. Identify atom runs in the model (contiguous blocks of the same code).
    let runs = identify_runs(&ts.model);

    // 2. Determine which runs fall within the region.
    let (region_start, region_end) = find_region_bounds(&runs, ts, region);

    // 3. Classify atoms and build helix/strand structures.
    let region_runs: Vec<_> = runs
        .iter()
        .filter(|r| r.start_col >= region_start && r.end_col() <= region_end)
        .cloned()
        .collect();

    // Identify helices: codes that appear exactly twice in the region.
    let mut code_runs: std::collections::BTreeMap<i32, Vec<&Run>> =
        std::collections::BTreeMap::new();
    for run in &region_runs {
        code_runs.entry(run.code).or_default().push(run);
    }

    let mut helices = Vec::new();
    let mut strands = Vec::new();
    let mut atoms = Vec::new();

    // First pass: build helices and strands.
    let mut helix_map: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();
    let mut strand_map: std::collections::BTreeMap<i32, usize> = std::collections::BTreeMap::new();

    for (&code, runs_for_code) in &code_runs {
        if runs_for_code.len() == 2 {
            // Helix: first run is 5', second is 3'.
            let run5 = runs_for_code[0];
            let run3 = runs_for_code[1];
            let helix_len = run5.num_cols;
            assert_eq!(helix_len, run3.num_cols, "helix strands must have equal length");

            // Compute gap statistics for the region between the helix strands.
            let inter_max_gaps = compute_inter_helix_gaps(ts, run5, run3, &region_runs);
            let max_dist = run3.start_col - run5.start_col - helix_len;
            let min_dist = max_dist - inter_max_gaps;

            let profile = profile::build_helix_profile(
                ts,
                &(run5.start_col..run5.start_col + helix_len).collect::<Vec<_>>(),
                &(run3.start_col..run3.start_col + helix_len).collect::<Vec<_>>(),
                bg,
                pseudo_count_weight,
                log_zero,
            );

            let hx_idx = helices.len();
            helix_map.insert(code, hx_idx);
            let profile_f32 = flatten_profile_f32(&profile, helix_len);
            // Stride 32 = next power of two ≥ DINUC_CODES (26). Pads the 6
            // unused rows with 0.0 so the SIMD inner loop never reads them.
            let profile_f32_t = flatten_profile_f32_t(&profile, helix_len, 32);
            let pair_table = build_helix_pair_table(&profile, helix_len);
            // Helices score over a fixed `helix_len` columns regardless of
            // gap configuration (the gap is *between* the two strands, not
            // within them), so min == max here.
            let upper_bound = column_max_sum(&profile, helix_len, helix_len);
            helices.push(Helix {
                id: code,
                helix_len,
                min_len: 2 * helix_len + min_dist,
                max_len: 2 * helix_len + max_dist,
                max_gaps: inter_max_gaps,
                min_dist,
                max_dist,
                db_begin_5p: run5.start_col,
                db_begin_3p: run3.start_col,
                min_bgn: 0, // set below
                profile,
                profile_f32,
                profile_f32_t,
                pair_table,
                upper_bound,
            });
        } else if runs_for_code.len() == 1 {
            let run = runs_for_code[0];
            let max_gaps = compute_strand_gaps(ts, run);
            let cols: Vec<usize> = (run.start_col..run.start_col + run.num_cols).collect();
            let profile =
                profile::build_strand_profile(ts, &cols, bg, pseudo_count_weight, log_zero);

            let st_idx = strands.len();
            strand_map.insert(code, st_idx);
            let profile_f32 = flatten_profile_f32(&profile, run.num_cols);
            // Stride 8 = next power of two ≥ NT_CODES (6). Pads codes 6..7 with
            // zero so the inner loop indexes without bounds checks.
            let profile_f32_t = flatten_profile_f32_t(&profile, run.num_cols, 8);
            // Strands score over `min_len + gap` columns, with gap in
            // `[0, max_gaps]`. The trailing columns can have negative max
            // values, so the true upper bound is the max prefix sum over
            // valid lengths, not the full sum to `num_cols`.
            let strand_min_len = run.num_cols - max_gaps;
            let upper_bound = column_max_sum(&profile, strand_min_len, run.num_cols);
            strands.push(Strand {
                id: code,
                min_len: run.num_cols - max_gaps,
                max_len: run.num_cols,
                max_gaps,
                db_begin: run.start_col,
                min_bgn: 0, // set below
                profile,
                profile_f32,
                profile_f32_t,
                upper_bound,
            });
        }
    }

    // Second pass: build atom list in left-to-right order and compute min_bgn.
    let mut cumulative_min_len = 0usize;
    let pattern_db_begin = region_runs.first().map_or(0, |r| r.start_col);

    for run in &region_runs {
        if let Some(&hx_idx) = helix_map.get(&run.code) {
            let is_first = code_runs[&run.code][0].start_col == run.start_col;
            if is_first {
                // HLX1: the helix starts here.
                helices[hx_idx].min_bgn = cumulative_min_len;
                let paired_run = code_runs[&run.code][1];
                let paired_atom_idx = region_runs
                    .iter()
                    .position(|r| r.start_col == paired_run.start_col)
                    .unwrap();
                atoms.push(Atom {
                    code: run.code,
                    atom_type: AtomType::Helix1,
                    element_index: hx_idx,
                    paired_atom: Some(paired_atom_idx),
                    db_begin: run.start_col,
                    num_columns: run.num_cols,
                    min_len: run.num_cols, // helix strand has no internal gaps
                    max_len: run.num_cols,
                    max_gaps: 0,
                });
                cumulative_min_len += run.num_cols;
            } else {
                // HLX2: the helix 3' strand.
                atoms.push(Atom {
                    code: run.code,
                    atom_type: AtomType::Helix2,
                    element_index: hx_idx,
                    paired_atom: None,
                    db_begin: run.start_col,
                    num_columns: run.num_cols,
                    min_len: run.num_cols,
                    max_len: run.num_cols,
                    max_gaps: 0,
                });
                cumulative_min_len += run.num_cols;
            }
        } else if let Some(&st_idx) = strand_map.get(&run.code) {
            strands[st_idx].min_bgn = cumulative_min_len;
            let max_gaps = strands[st_idx].max_gaps;
            atoms.push(Atom {
                code: run.code,
                atom_type: AtomType::Strand,
                element_index: st_idx,
                paired_atom: None,
                db_begin: run.start_col,
                num_columns: run.num_cols,
                min_len: run.num_cols - max_gaps,
                max_len: run.num_cols,
                max_gaps,
            });
            cumulative_min_len += run.num_cols - max_gaps;
        }
    }

    // Compute pattern total lengths.
    let min_len = cumulative_min_len;
    let max_len: usize = region_runs.iter().map(|r| r.num_cols).sum();

    Pattern {
        atoms,
        helices,
        strands,
        min_len,
        max_len,
        db_begin: pattern_db_begin,
    }
}

/// A contiguous run of columns with the same model code.
#[derive(Debug, Clone)]
struct Run {
    code: i32,
    start_col: usize,
    num_cols: usize,
}

impl Run {
    fn end_col(&self) -> usize {
        self.start_col + self.num_cols
    }
}

/// Identify contiguous runs of the same code in the model.
fn identify_runs(model: &[i32]) -> Vec<Run> {
    let mut runs = Vec::new();
    if model.is_empty() {
        return runs;
    }

    let mut start = 0;
    let mut current_code = model[0];

    for (i, &code) in model.iter().enumerate().skip(1) {
        if code != current_code {
            runs.push(Run {
                code: current_code,
                start_col: start,
                num_cols: i - start,
            });
            start = i;
            current_code = code;
        }
    }
    runs.push(Run {
        code: current_code,
        start_col: start,
        num_cols: model.len() - start,
    });

    runs
}

/// Find the alignment column range for the specified region.
///
/// Region (begin, end): negative values indicate the first run of a helix code,
/// positive values indicate the second run (or only run for strands).
fn find_region_bounds(runs: &[Run], ts: &TrainingSet, region: &Region) -> (usize, usize) {
    let begin_code = region.begin.unsigned_abs() as i32;
    let end_code = region.end.unsigned_abs() as i32;

    // Find the target run for 'begin'.
    let begin_run = if region.begin < 0 {
        // Negative: first run of this helix code.
        runs.iter()
            .find(|r| r.code == begin_code && ts.helix_codes.contains(&r.code))
    } else {
        // Positive: second run of helix code, or only run of strand code.
        if ts.helix_codes.contains(&end_code) {
            runs.iter()
                .filter(|r| r.code == begin_code)
                .nth(0) // For begin, use first occurrence even if positive
        } else {
            runs.iter().find(|r| r.code == begin_code)
        }
    };

    let end_run = if region.end > 0 && ts.helix_codes.contains(&end_code) {
        // Positive helix code: second run.
        runs.iter().filter(|r| r.code == end_code).nth(1)
    } else {
        runs.iter().filter(|r| r.code == end_code).last()
    };

    let start = begin_run.map_or(0, |r| r.start_col);
    let end = end_run.map_or(ts.alignment_len, |r| r.end_col());

    (start, end)
}

/// Count the total max_gaps of atoms between a helix's 5' and 3' strands.
fn compute_inter_helix_gaps(ts: &TrainingSet, run5: &Run, run3: &Run, all_runs: &[Run]) -> usize {
    let mut total_gaps = 0;
    for run in all_runs {
        if run.start_col >= run5.end_col() && run.end_col() <= run3.start_col {
            total_gaps += compute_strand_gaps(ts, run);
        }
    }
    total_gaps
}

/// Compute `max_gaps` for an atom run, matching `libsrc/atom.c::ReadAtoms`:
/// the maximum number of gaps any single training sequence carries within the
/// atom's columns. (The previous implementation counted the number of *columns*
/// containing at least one gap, which inflated `max_gaps` when gaps were
/// distributed across sequences and produced a different config space than C.)
fn compute_strand_gaps(ts: &TrainingSet, run: &Run) -> usize {
    let mut max_gaps = 0;
    for seq in &ts.sequences {
        let gaps = (run.start_col..run.end_col())
            .filter(|&col| seq[col] == b'-')
            .count();
        if gaps > max_gaps {
            max_gaps = gaps;
        }
    }
    max_gaps
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::epn;

    #[test]
    fn test_build_trna_pattern() {
        let ts = epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let region = Region {
            begin: -2,
            end: 2,
        };
        let bg = Background::default();
        let pattern = build_pattern(&ts, &region, &bg, 0.0002, -20.0);

        // tRNA type I with region -2,2 should have 4 helices and 6 strands.
        assert_eq!(
            pattern.helices.len(),
            4,
            "expected 4 helices, got {}",
            pattern.helices.len()
        );
        assert_eq!(
            pattern.strands.len(),
            6,
            "expected 6 strands, got {}",
            pattern.strands.len()
        );

        // 14 atoms: 4 helices × 2 strands + 6 isolated strands.
        assert_eq!(pattern.atoms.len(), 14);

        // Helix 2 (outermost) should have helix_len=7.
        let h2 = pattern.helices.iter().find(|h| h.id == 2).unwrap();
        assert_eq!(h2.helix_len, 7);
    }
}
