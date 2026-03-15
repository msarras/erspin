# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**erspin** is a Rust rewrite of ERPIN (Easy RNA Profile Identification), a bioinformatics tool that searches for RNA motifs in nucleotide sequence databases.

ERPIN works by taking a training set alignment (`.epn` file) and scanning FASTA sequence databases for matching RNA structural motifs using multi-level mask processing with configurable score thresholds/cutoffs.

## Build & Run

```bash
cargo build          # debug build
cargo run            # run
cargo test           # run all tests
cargo test <name>    # run a single test by name
cargo bench          # run all criterion benchmarks
cargo bench -- "search_full"  # run specific benchmark group
```

Rust edition: 2024 (requires Rust 1.85+).

## Rust Codebase Architecture

Single binary with subcommands (`search`, `view`, `stats`, `eval`, `configs`), using clap derive.

### Modules

- `src/main.rs` — CLI entry point, subcommand definitions, argument parsing
- `src/lib.rs` — library root re-exporting all modules
- `src/types.rs` — core data structures: `TrainingSet`, `Sequence`, `Pattern`, `Atom`, `Helix`, `Strand`, `Mask`, `Region`, `Hit`, and enums for mask modes, strand directions, output styles
- `src/epn.rs` — `.epn` training set parser (model digit lines → element codes, helix/strand identification by occurrence count)
- `src/fasta.rs` — streaming FASTA reader (`FastaReader`) that yields one `Sequence` at a time
- `src/profile.rs` — log-odds scoring profile builder for strands (6×len) and helices (24×len), background frequency computation. Uses precomputed LUTs (`NT_STRAND_LUT`, `DINUC_LUT`) for encoding in hot loops.
- `src/region.rs` — region spec parser (`-2,2` format) and mask spec parser (`!6,8 / 2,3 / *` format)
- `src/pattern.rs` — pattern builder: identifies atom runs, builds helices/strands with gap statistics and profiles from training set
- `src/scoring.rs` — score table computation (delegates to simd.rs), `ConfigLookup` struct with suffix-max bounds for fast config scoring with early termination
- `src/simd.rs` — optimized scoring routines: column-major loop reorder for strands/helices, interleaved multi-position DP for gapped strands. All produce identical results to scalar code (verified by tests).
- `src/search.rs` — mask resolution, threshold computation, config generation, multi-level cascade search, Rayon parallel search, overlap filtering
- `src/output.rs` — structured output formatting: text (default), JSON, TSV; output styles: full, compact, quiet
- `src/evalue.rs` — E-value computation via convolution-based histogram method: per-column score distributions → element histograms → mask histogram → CDF → extreme value correction
- `src/error.rs` — error types with `thiserror`
- `benches/search_bench.rs` — criterion benchmarks: component (helix/strand/config scoring), full search at various sizes, CLI throughput

### Key Design Decisions

- Profiles/masks are immutable during search → safe to share across threads
- FASTA reader is streaming to support multi-GB databases
- Mask spec syntax redesigned: `!` for umask, `+` for add, `*` for nomask, `/` separates levels
- Region args use `allow_hyphen_values` since specs like `-2,2` start with hyphen
- Config generation: only strand atoms are gap variables; helix gaps are derived from intervening atom positions
- Multi-level search window uses `pattern.max_len` extent (not mask-specific), since different masks start at different atoms
- Gapped strand DP uses pre-allocated flat buffer with banded reset (only touches band entries, not full table)
- `ConfigLookup` caches slice references and uses suffix-max bounds for early termination in the config scoring inner loop

### Performance Characteristics

- Helix score tables: ~14-73 MiB/s (column-major with f32 profile lookup)
- No-gap strand score tables: ~120-135 MiB/s (column-major accumulation)
- Gapped strand DP: ~5-80 MiB/s depending on gap count (interleaved multi-position banded DP)
- Full search throughput: ~3.3 MiB/s at 1MB sequences
- Rust vs C ERPIN: ~18x faster at 10KB, ~4x at 100KB, ~1.7x at 1MB

## Key ERPIN Concepts

- **Training set (.epn)**: Two model digit lines encode secondary structure (read column-by-column, concatenated vertically to form element codes). Codes appearing twice → helix (5'/3' strands); once → isolated strand.
- **Masks**: Subsets of structural elements used at each search level.
- **Multi-level search**: Sequential filtering — each level applies masks with cutoff thresholds, narrowing candidates. Level N+1 only processes hits from level N.
- **Scoring**: Log-odds weight matrices. Helices use dinucleotide (base-pair) codes (24 values); strands use single-nucleotide codes (6 values).
- **Static vs Dynamic mask processing**: Static precomputes all gap configurations; Dynamic recomputes per-hit using winning config from previous level.
