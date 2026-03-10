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
        #[arg(short = 't', long)]
        training_set: String,

        /// Database file (FASTA format).
        #[arg(short = 'd', long)]
        database: String,

        /// Region specification (e.g., -2,2).
        #[arg(short = 'r', long, allow_hyphen_values = true)]
        region: String,

        /// Mask levels, separated by '/'. Use '*' for all remaining elements,
        /// '!' prefix for umask, '+' prefix to add to previous level.
        /// Example: "6,8 / !2,3 / *"
        #[arg(short = 'l', long)]
        levels: String,

        /// Cutoff thresholds per level, comma-separated.
        /// Values can be percentages (e.g., "100%") or raw scores.
        #[arg(short = 'c', long, default_value = "100%")]
        cutoffs: String,

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
    },

    /// View training set structure and alignment.
    View {
        /// Training set file (.epn format).
        #[arg(short = 't', long)]
        training_set: String,

        /// Region specification (e.g., -2,2).
        #[arg(short = 'r', long, allow_hyphen_values = true)]
        region: Option<String>,
    },

    /// Show score statistics for the training set.
    Stats {
        /// Training set file (.epn format).
        #[arg(short = 't', long)]
        training_set: String,

        /// Region specification.
        #[arg(short = 'r', long, allow_hyphen_values = true)]
        region: String,

        /// Mask levels.
        #[arg(short = 'l', long)]
        levels: String,
    },

    /// Calculate E-values for given cutoffs.
    Eval {
        /// Training set file (.epn format).
        #[arg(short = 't', long)]
        training_set: String,

        /// Region specification.
        #[arg(short = 'r', long, allow_hyphen_values = true)]
        region: String,

        /// Mask levels.
        #[arg(short = 'l', long)]
        levels: String,

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
        #[arg(short = 't', long)]
        training_set: String,

        /// Region specification.
        #[arg(short = 'r', long, allow_hyphen_values = true)]
        region: String,

        /// Mask levels.
        #[arg(short = 'l', long)]
        levels: String,
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

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Search {
            training_set,
            database,
            region: region_str,
            levels,
            cutoffs,
            processing: _processing,
            strand,
            output: output_style,
            background: _background,
            pseudo_count,
            format: output_format,
        } => {
            // Parse inputs.
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;

            let reg = region::parse_region(&region_str)
                .with_context(|| format!("parsing region '{}'", region_str))?;

            let mask_specs = region::parse_mask_specs(&levels)
                .with_context(|| format!("parsing mask levels '{}'", levels))?;

            // Build pattern with profiles.
            let bg = Background::default();
            let pat = pattern::build_pattern(&ts, &reg, &bg, pseudo_count, -20.0);

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

            // Resolve masks and compute thresholds.
            let mut masks = search::resolve_masks(&mask_specs, &pat);

            let cutoff_list: Vec<String> = cutoffs.split(',').map(|s| s.trim().to_string()).collect();
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

            // Load database and search.
            let reader = FastaReader::from_path(&database)
                .with_context(|| format!("opening database '{}'", database))?;
            let sequences = reader
                .collect_all()
                .with_context(|| format!("reading database '{}'", database))?;

            let total_bases: usize = sequences.iter().map(|s| s.len()).sum();
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
            levels,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = region::parse_mask_specs(&levels)?;
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
            levels,
            megabases,
            both_strands,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = region::parse_mask_specs(&levels)?;
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
            levels,
        } => {
            let ts = epn::parse_epn(&training_set)
                .with_context(|| format!("loading training set '{}'", training_set))?;
            let reg = region::parse_region(&region_str)?;
            let mask_specs = region::parse_mask_specs(&levels)?;
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
