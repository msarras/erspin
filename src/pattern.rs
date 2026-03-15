use crate::profile::{self, Background};
use crate::types::*;

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
            });
        } else if runs_for_code.len() == 1 {
            let run = runs_for_code[0];
            let max_gaps = compute_strand_gaps(ts, run);
            let cols: Vec<usize> = (run.start_col..run.start_col + run.num_cols).collect();
            let profile =
                profile::build_strand_profile(ts, &cols, bg, pseudo_count_weight, log_zero);

            let st_idx = strands.len();
            strand_map.insert(code, st_idx);
            strands.push(Strand {
                id: code,
                min_len: run.num_cols - max_gaps,
                max_len: run.num_cols,
                max_gaps,
                db_begin: run.start_col,
                min_bgn: 0, // set below
                profile,
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

/// Count the number of variable gap columns in a strand run.
/// A column is a "gap column" if at least one training sequence has a gap there
/// but not all sequences do (it's variable).
fn compute_strand_gaps(ts: &TrainingSet, run: &Run) -> usize {
    let mut gap_cols = 0;
    for col in run.start_col..run.end_col() {
        let gap_count = ts
            .sequences
            .iter()
            .filter(|seq| seq[col] == b'-')
            .count();
        // Variable gap: some have gaps, some don't. Or all have gaps (void column).
        if gap_count > 0 {
            gap_cols += 1;
        }
    }
    gap_cols
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
