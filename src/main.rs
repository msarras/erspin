use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::time::Instant;

use erspin::epn;
use erspin::evalue;
use erspin::fasta::FastaReader;
use erspin::output;
use erspin::pattern;
use erspin::profile::Background;
use erspin::region;
use erspin::search;
use erspin::types::*;

#[derive(Parser)]
#[command(name = "erspin", version, about = "RNA motif search using profile-based methods")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Search for RNA motifs in a sequence database.
    Search {
        /// Training set file (.epn format).
        training_set: String,

        /// Database file (FASTA format).
        database: String,

        /// Region specification (e.g., 1,23 or -2,2).
        #[arg(allow_hyphen_values = true)]
        region: String,

        /// Mask level elements (space- or comma-separated). Each --add starts a new level.
        /// First group is the initial mask, subsequent groups add elements cumulatively.
        /// Example: --add 2 4 5 6 --add 3 8 9 10  (or --add 2,4,5,6 --add 3,8,9,10)
        #[arg(long = "add")]
        add: Vec<String>,

        /// Cutoff thresholds per level (raw scores or percentages like 100%).
        /// Example: --cutoff 10 15 30
        #[arg(long, num_args = 1..)]
        cutoff: Option<Vec<String>>,

        /// Minimum log-odds value for zero-count positions.
        #[arg(long, default_value = "-20.0", allow_hyphen_values = true)]
        logzero: f64,

        /// Mask processing strategy.
        #[arg(long, default_value = "dynamic", value_parser = parse_mask_processing)]
        processing: MaskProcessing,

        /// Strand direction to search.
        #[arg(long, default_value = "both", value_parser = parse_strand_direction)]
        strand: StrandDirection,

        /// Output style.
        #[arg(long, default_value = "full", value_parser = parse_output_style)]
        output: OutputStyle,

        /// Background statistics mode.
        #[arg(long, default_value = "global", value_parser = parse_background_mode)]
        background: BackgroundMode,

        /// Pseudo-count weight (0.0 to 1.0).
        #[arg(long, default_value = "0.0002")]
        pseudo_count: f64,

        /// Output format.
        #[arg(long, default_value = "text", value_parser = ["text", "json", "tsv"])]
        format: String,

        /// Number of threads to use for parallel search.
        #[arg(long, default_value = "4")]
        cpu: usize,
    },

    /// View training set structure and alignment.
    View {
        /// Training set file (.epn format).
        training_set: String,

        /// Region specification (e.g., -2,2).
        #[arg(allow_hyphen_values = true)]
        region: Option<String>,
    },

    /// Show score statistics for the training set.
    Stats {
        /// Training set file (.epn format).
        training_set: String,

        /// Region specification.
        #[arg(allow_hyphen_values = true)]
        region: String,

        /// Mask level elements (space- or comma-separated). Each --add starts a new level.
        #[arg(long = "add")]
        add: Vec<String>,
    },

    /// Calculate E-values for given cutoffs.
    Eval {
        /// Training set file (.epn format).
        training_set: String,

        /// Region specification.
        #[arg(allow_hyphen_values = true)]
        region: String,

        /// Mask level elements (space- or comma-separated). Each --add starts a new level.
        #[arg(long = "add")]
        add: Vec<String>,

        /// Database size in megabases (default: 1.0).
        #[arg(short = 'm', long, default_value = "1.0")]
        megabases: f64,

        /// Search both strands (doubles effective database size).
        #[arg(long, default_value = "true")]
        both_strands: bool,
    },

    /// Estimate memory usage and configuration counts.
    Configs {
        /// Training set file (.epn format).
        training_set: String,

        /// Region specification.
        #[arg(allow_hyphen_values = true)]
        region: String,

        /// Mask level elements (space- or comma-separated). Each --add starts a new level.
        #[arg(long = "add")]
        add: Vec<String>,
    },
}

fn parse_mask_processing(s: &str) -> Result<MaskProcessing, String> {
    match s {
        "dynamic" | "dmp" => Ok(MaskProcessing::Dynamic),
        "static" | "smp" => Ok(MaskProcessing::Static),
        _ => Err(format!(
            "invalid processing mode: '{}' (expected: dynamic, static)",
            s
        )),
    }
}

fn parse_strand_direction(s: &str) -> Result<StrandDirection, String> {
    match s {
        "forward" | "fwd" => Ok(StrandDirection::Forward),
        "reverse" | "rev" => Ok(StrandDirection::Reverse),
        "both" => Ok(StrandDirection::Both),
        _ => Err(format!(
            "invalid strand direction: '{}' (expected: forward, reverse, both)",
            s
        )),
    }
}

fn parse_output_style(s: &str) -> Result<OutputStyle, String> {
    match s {
        "full" => Ok(OutputStyle::Full),
        "compact" => Ok(OutputStyle::Compact),
        "quiet" => Ok(OutputStyle::Quiet),
        _ => Err(format!(
            "invalid output style: '{}' (expected: full, compact, quiet)",
            s
        )),
    }
}

fn parse_background_mode(s: &str) -> Result<BackgroundMode, String> {
    match s {
        "global" => Ok(BackgroundMode::Global),
        "local" => Ok(BackgroundMode::Local),
        "uniform" => Ok(BackgroundMode::Uniform),
        _ => Err(format!(
            "invalid background mode: '{}' (expected: global, local, uniform)",
            s
        )),
    }
}

/// Compute ATGC background frequencies across the concatenated database.
///
/// Matches C ERPIN's behaviour: counts A/T/G/C across all bases (case-insensitive)
/// and reports the relative frequency. Falls back to uniform when no scoreable
/// bases are present.
fn background_from_sequences(sequences: &[Sequence]) -> Background {
    let mut counts = [0u64; 4];
    let mut total = 0u64;
    for seq in sequences {
        for &b in &seq.data {
            let idx = match b {
                b'A' | b'a' => Some(0),
                b'T' | b't' | b'U' | b'u' => Some(1),
                b'G' | b'g' => Some(2),
                b'C' | b'c' => Some(3),
                _ => None,
            };
            if let Some(i) = idx {
                counts[i] += 1;
                total += 1;
            }
        }
    }
    if total == 0 {
        return Background::default();
    }
    Background {
        freq: [
            counts[0] as f64 / total as f64,
            counts[1] as f64 / total as f64,
            counts[2] as f64 / total as f64,
            counts[3] as f64 / total as f64,
        ],
    }
}

/// Convert --add groups into MaskSpec entries.
/// Each --add value is a comma-separated list of element numbers
/// (space-separated tokens are folded into the same group by `coalesce_add_argv`
/// before clap parses them).
/// First group → Mask mode (specific elements), subsequent → Add mode (cumulative).
/// If no groups provided, defaults to NoMask (all elements).
fn add_groups_to_mask_specs(add: &[String]) -> Result<Vec<MaskSpec>> {
    if add.is_empty() {
        return Ok(vec![MaskSpec {
            mode: MaskMode::NoMask,
            elements: Vec::new(),
        }]);
    }

    let mut specs = Vec::new();
    for (i, group_str) in add.iter().enumerate() {
        let elements: Vec<usize> = group_str
            .split(',')
            .map(|s| {
                s.trim()
                    .parse::<usize>()
                    .with_context(|| format!("invalid element '{}' in --add", s.trim()))
            })
            .collect::<Result<Vec<_>>>()?;
        let mode = if i == 0 {
            MaskMode::Mask
        } else {
            MaskMode::Add
        };
        specs.push(MaskSpec { mode, elements });
    }
    Ok(specs)
}

/// Preprocess argv so that space-separated values after `--add` are folded
/// into a single comma-separated value, matching the legacy `--add a,b,c`
/// shape that clap's derive expects.
///
/// `--add 5 6 7 --add 4 8 10 12 13 --logzero -3 --cutoff 4 8 10 20`
///   → `--add 5,6,7 --add 4,8,10,12,13 --logzero -3 --cutoff 4 8 10 20`
///
/// Already-comma-separated input passes through unchanged. Only `--add`
/// (long form) is rewritten — the boundary is the next argv token that
/// starts with `-` (i.e. another flag) or end-of-argv. Element numbers are
/// always positive integers, so a leading `-` unambiguously marks a flag.
fn coalesce_add_argv<I, S>(args: I) -> Vec<String>
where
    I: IntoIterator<Item = S>,
    S: Into<String>,
{
    let mut out = Vec::new();
    let mut iter = args.into_iter().map(Into::into).peekable();
    while let Some(arg) = iter.next() {
        if arg == "--add" {
            out.push(arg);
            let mut group: Vec<String> = Vec::new();
            while let Some(next) = iter.peek() {
                if next.starts_with('-') {
                    break;
                }
                group.push(iter.next().unwrap());
            }
            // If user wrote `--add` with no following values, push nothing
            // and let clap report the missing-value error itself.
            if !group.is_empty() {
                out.push(group.join(","));
            }
        } else {
            out.push(arg);
        }
    }
    out
}

#[cfg(test)]
mod argv_tests {
    use super::coalesce_add_argv;

    fn run(args: &[&str]) -> Vec<String> {
        coalesce_add_argv(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn space_separated_groups_collapse_to_commas() {
        assert_eq!(
            run(&[
                "erspin", "search", "x.epn", "db.fa", "1,24",
                "--add", "5", "6", "7",
                "--add", "4", "8", "10", "12", "13",
                "--logzero", "-3",
                "--cutoff", "4", "8", "10", "20",
            ]),
            vec![
                "erspin", "search", "x.epn", "db.fa", "1,24",
                "--add", "5,6,7",
                "--add", "4,8,10,12,13",
                "--logzero", "-3",
                "--cutoff", "4", "8", "10", "20",
            ],
        );
    }

    #[test]
    fn comma_separated_input_passes_through() {
        assert_eq!(
            run(&["--add", "5,6,7", "--add", "4,8"]),
            vec!["--add", "5,6,7", "--add", "4,8"],
        );
    }

    #[test]
    fn mixed_form_concatenates_with_commas() {
        // `--add 5 6,7 --add 4` → `--add 5,6,7 --add 4`
        assert_eq!(
            run(&["--add", "5", "6,7", "--add", "4"]),
            vec!["--add", "5,6,7", "--add", "4"],
        );
    }

    #[test]
    fn empty_add_passes_through_for_clap_to_error() {
        // `--add --logzero -3` — no values; we don't synthesise one.
        assert_eq!(
            run(&["--add", "--logzero", "-3"]),
            vec!["--add", "--logzero", "-3"],
        );
    }

    #[test]
    fn non_add_args_untouched() {
        assert_eq!(
            run(&["--cutoff", "4", "8", "10", "20"]),
            vec!["--cutoff", "4", "8", "10", "20"],
        );
    }
}

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse_from(coalesce_add_argv(std::env::args()));

    match cli.command {
        Commands::Search {
            training_set,
            database,
            region: region_str,
            add,
            cutoff,
            logzero,
            processing: _processing,
            strand,
            output: output_style,
            background,
            pseudo_count,
            format: output_format,
            cpu,
        } => {
            rayon::ThreadPoolBuilder::new()
                .num_threads(cpu)
                .build_global()
                .context("configuring thread pool")?;
            // Parse inputs.
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;

            let reg = region::parse_region(&region_str)
                .with_context(|| format!("parsing region '{}'", region_str))?;

            let mask_specs = add_groups_to_mask_specs(&add)
                .context("parsing --add mask levels")?;

            // Load database first so we can compute database-wide background
            // frequencies (matches C ERPIN's "ATGC ratios" computation).
            let reader = FastaReader::from_path(&database)
                .with_context(|| format!("opening database '{}'", database))?;
            let sequences = reader
                .collect_all()
                .with_context(|| format!("reading database '{}'", database))?;
            let total_bases: usize = sequences.iter().map(|s| s.len()).sum();

            // Build pattern with profiles.
            let bg = match background {
                BackgroundMode::Uniform => Background::default(),
                BackgroundMode::Global | BackgroundMode::Local => {
                    background_from_sequences(&sequences)
                }
            };
            let pat = pattern::build_pattern(&ts, &reg, &bg, pseudo_count, logzero);

            eprintln!(
                "Training set: {:?}\n  {} sequences of length {}",
                training_set, ts.nseq, ts.alignment_len
            );
            eprintln!(
                "  {} helices, {} strands ({} atoms)",
                pat.helices.len(),
                pat.strands.len(),
                pat.atoms.len()
            );
            eprintln!(
                "  ATGC background: {:.3}  {:.3}  {:.3}  {:.3}",
                bg.freq[0], bg.freq[1], bg.freq[2], bg.freq[3]
            );

            // Resolve masks and compute thresholds.
            let mut masks = search::resolve_masks(&mask_specs, &pat);

            let cutoff_list: Vec<String> = match cutoff {
                Some(vals) => vals,
                None => vec!["100%".to_string()],
            };
            search::compute_thresholds(&ts, &pat, &mut masks, &cutoff_list);

            for (i, mask) in masks.iter().enumerate() {
                eprintln!(
                    "  Level {}: {} helices + {} strands, {} configs, cutoff {:.2}",
                    i + 1,
                    mask.hx_indices.len(),
                    mask.st_indices.len(),
                    mask.configs.len(),
                    mask.threshold
                );
            }

            eprintln!(
                "Database: {:?}\n  {} nucleotides in {} sequence(s)",
                database, total_bases, sequences.len()
            );

            let start_time = Instant::now();
            let mut results = search::search_all_parallel(&pat, &masks, &sequences, strand);
            let elapsed = start_time.elapsed();

            // Annotate hits with E-values from the final mask.
            let datavol = total_bases as f64
                * if matches!(strand, StrandDirection::Both) {
                    2.0
                } else {
                    1.0
                };
            let final_mask = masks.last().unwrap();
            for (_seq_idx, hits) in &mut results {
                evalue::annotate_hits(hits, &pat, final_mask, &bg, datavol);
            }

            output::write_results(
                &results,
                &sequences,
                &output_format,
                output_style,
            )?;

            let total_hits: usize = results.iter().map(|(_, hits)| hits.len()).sum();
            eprintln!(
                "\n{} hits found in {:.3}s.",
                total_hits,
                elapsed.as_secs_f64()
            );
        }

        Commands::View {
            training_set,
            region: region_str,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;

            println!(
                "Training set: {:?}\n  {} sequences of length {}",
                training_set, ts.nseq, ts.alignment_len
            );
            println!(
                "  {} helices, {} strands ({} atoms total)",
                ts.nhelix, ts.nstrand, ts.natom
            );
            println!("\nElement codes (left-to-right): {:?}", ts.atom_list);
            println!("Helix codes: {:?}", ts.helix_codes);
            println!("Strand codes: {:?}", ts.strand_codes);

            print!("\nModel: ");
            let mut prev = -1;
            for &code in &ts.model {
                if code != prev {
                    print!("[{}]", code);
                    prev = code;
                }
            }
            println!();

            print!("       ");
            for &code in &ts.model {
                if ts.helix_codes.contains(&code) {
                    print!("H");
                } else {
                    print!("S");
                }
            }
            println!();

            if let Some(ref region_str) = region_str {
                let reg = region::parse_region(region_str)
                    .with_context(|| format!("parsing region '{}'", region_str))?;
                let bg = Background::default();
                let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);
                println!(
                    "\nRegion {}: {} helices, {} strands, {} atoms",
                    region_str,
                    pat.helices.len(),
                    pat.strands.len(),
                    pat.atoms.len()
                );
                println!(
                    "  Pattern length: {}..{} (min..max)",
                    pat.min_len, pat.max_len
                );
            }

            let show_count = ts.nseq.min(15);
            println!("\nFirst {} sequences:", show_count);
            for i in 0..show_count {
                let seq_str: String = ts.sequences[i].iter().map(|&b| b as char).collect();
                println!("{:<4} {}", i + 1, seq_str);
            }
        }

        Commands::Stats {
            training_set,
            region: region_str,
            add,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = add_groups_to_mask_specs(&add)
                .context("parsing --add mask levels")?;
            let bg = Background::default();
            let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);
            let masks = search::resolve_masks(&mask_specs, &pat);

            println!(
                "Training set: {} sequences of length {}",
                ts.nseq, ts.alignment_len
            );
            println!(
                "  {} helices, {} strands ({} atoms)",
                pat.helices.len(),
                pat.strands.len(),
                pat.atoms.len()
            );
            println!(
                "  Pattern length: {}..{} (min..max)",
                pat.min_len, pat.max_len
            );

            for (i, mask) in masks.iter().enumerate() {
                println!(
                    "\nLevel {}: {} helices + {} strands, {} configurations",
                    i + 1,
                    mask.hx_indices.len(),
                    mask.st_indices.len(),
                    mask.configs.len()
                );

                // Score all training sequences with this mask.
                let mut scores: Vec<f64> = ts
                    .sequences
                    .iter()
                    .map(|seq| {
                        erspin::scoring::score_training_sequence(seq, &pat, mask)
                    })
                    .collect();
                scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let min = scores.first().copied().unwrap_or(0.0);
                let max = scores.last().copied().unwrap_or(0.0);
                let mean = scores.iter().sum::<f64>() / scores.len().max(1) as f64;
                let median = scores[scores.len() / 2];

                println!("  Training scores: min={:.2} median={:.2} mean={:.2} max={:.2}", min, median, mean, max);
                println!("  100% cutoff: {:.2}", min);
                if scores.len() >= 10 {
                    let idx90 = ((scores.len() as f64) * 0.1).ceil() as usize;
                    println!("   90% cutoff: {:.2}", scores[idx90]);
                }
            }
        }

        Commands::Eval {
            training_set,
            region: region_str,
            add,
            megabases,
            both_strands,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = add_groups_to_mask_specs(&add)
                .context("parsing --add mask levels")?;
            let bg = Background::default();
            let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);
            let mut masks = search::resolve_masks(&mask_specs, &pat);

            // Use 100% cutoff for all levels to get the threshold.
            let cutoffs: Vec<String> = (0..masks.len()).map(|_| "100%".to_string()).collect();
            search::compute_thresholds(&ts, &pat, &mut masks, &cutoffs);

            let datavol = megabases * 1e6 * if both_strands { 2.0 } else { 1.0 };
            let final_mask = masks.last().unwrap();
            let table = evalue::EvalueTable::build(&pat, final_mask, &bg, datavol);

            let strand_str = if both_strands {
                "double strand"
            } else {
                "single strand"
            };

            println!(
                "Training set: {} sequences of length {}",
                ts.nseq, ts.alignment_len
            );
            println!(
                "  {} helices, {} strands ({} atoms)",
                pat.helices.len(),
                pat.strands.len(),
                pat.atoms.len()
            );
            println!(
                "\nE-value at cutoff {:.1} for {:.1}Mb {} data: {:.2e}",
                final_mask.threshold,
                megabases,
                strand_str,
                table.evalue(final_mask.threshold)
            );

            // Show E-values at a few representative scores.
            println!("\nScore → E-value table:");
            let scores = [40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
            for &s in &scores {
                println!("  {:.0}\t{:.2e}", s, table.evalue(s));
            }
        }

        Commands::Configs {
            training_set,
            region: region_str,
            add,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = add_groups_to_mask_specs(&add)
                .context("parsing --add mask levels")?;
            let bg = Background::default();
            let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);
            let masks = search::resolve_masks(&mask_specs, &pat);

            println!(
                "Training set: {} sequences of length {}",
                ts.nseq, ts.alignment_len
            );
            for (i, mask) in masks.iter().enumerate() {
                println!(
                    "Level {}: {} helices + {} strands, {} configurations",
                    i + 1,
                    mask.hx_indices.len(),
                    mask.st_indices.len(),
                    mask.configs.len()
                );
            }
        }
    }

    Ok(())
}
