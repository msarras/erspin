# erspin

A Rust rewrite of [ERPIN](http://rssf.i2bc.paris-saclay.fr/Software/erpin.php) (Easy RNA Profile Identification), a bioinformatics tool that searches for RNA structural motifs in nucleotide sequence databases using profile-based scoring.

Given a training set alignment (`.epn` file) describing an RNA secondary structure, erspin scans FASTA databases for matching motifs using multi-level mask filtering with configurable score thresholds.

## Concepts

**Training sets** (`.epn` files) encode RNA secondary structure as a multiple sequence alignment. Two model digit lines define structural elements: codes appearing twice denote helices (base-paired 5'/3' strands), codes appearing once denote isolated strands.

**Scoring** uses log-odds weight matrices built from the training alignment. Helices use dinucleotide (base-pair) codes (24 values); strands use single-nucleotide codes (6 values). Gapped strands are scored via banded Needleman-Wunsch DP.

**Multi-level search** applies a cascade of masks (subsets of structural elements) with increasing completeness. Each level filters candidates using a score cutoff, so only positions passing level N are evaluated at level N+1. This dramatically reduces computation on large databases.

**Mask specification** syntax:
- Element indices (e.g., `6,8`) — use these elements at this level
- `!` prefix (umask) — use all elements *except* these
- `+` prefix — add elements to the previous level's mask
- `*` (nomask) — use all elements
- `/` separates levels

Example: `!6,8 / +2,3 / *` defines three levels — first uses everything except elements 6 and 8, second adds elements 2 and 3, third uses all elements.

## Installation

Requires Rust 1.85+ (edition 2024).

```bash
# Build optimized binary
cargo build --release

# Binary is at target/release/erspin
./target/release/erspin --help

# Or install into your PATH
cargo install --path .
```

## Usage

### search

Search for RNA motifs in a FASTA database:

```bash
erspin search \
  -t training.epn \
  -d database.fasta \
  -r "-2,2" \
  -l "!6,8 / +2,3 / *" \
  -c "100%,100%,90%"
```

| Flag | Description | Default |
|------|-------------|---------|
| `-t` | Training set file (`.epn` format) | required |
| `-d` | Database file (FASTA format) | required |
| `-r` | Region specification (e.g., `-2,2`) | required |
| `-l` | Mask levels, `/`-separated | required |
| `-c` | Cutoff thresholds per level, comma-separated (percentage or raw score) | `100%` |
| `--processing` | `dynamic` or `static` mask processing | `dynamic` |
| `--strand` | `forward`, `reverse`, or `both` | `both` |
| `--output` | `full`, `compact`, or `quiet` | `full` |
| `--background` | `global`, `local`, or `uniform` | `global` |
| `--pseudo-count` | Pseudo-count weight (0.0–1.0) | `0.0002` |
| `--format` | `text`, `json`, or `tsv` | `text` |

### view

Inspect a training set's structure and alignment:

```bash
erspin view -t training.epn
erspin view -t training.epn -r "-2,2"
```

### stats

Show score statistics (min/median/mean/max) for training sequences at each mask level:

```bash
erspin stats -t training.epn -r "-2,2" -l "6,8 / *"
```

### eval

Estimate E-values for given cutoffs and database size:

```bash
erspin eval -t training.epn -r "-2,2" -l "6,8 / *" -m 1.0
```

### configs

Show configuration counts per mask level:

```bash
erspin configs -t training.epn -r "-2,2" -l "6,8 / *"
```

## Quick start with test data

The repository includes tRNA test data from the original ERPIN distribution:

```bash
cargo build --release

./target/release/erspin search \
  -t erpin5.5.4.serv/start.test/trna.typeI.epn \
  -d erpin5.5.4.serv/start.test/test.trna.fasta \
  -r "-2,2" \
  -l "!6,8 / +2,3 / *" \
  -c "100%,100%,90%"
```

Expected reference output from the original C ERPIN is in `erpin5.5.4.serv/start.test/test.txt`.

## Testing

```bash
cargo test           # run all 25 tests
cargo test <name>    # run a single test by name
```

Tests cover EPN parsing, region/mask specification parsing, nucleotide encoding, profile construction, SIMD correctness (bit-identical verification against scalar baselines), search functionality, and E-value computation. Most tests use the real tRNA training set for integration-level validation.

## Benchmarks

Benchmarks use [Criterion](https://github.com/bheisler/criterion.rs) with synthetic DNA sequences:

```bash
cargo bench                       # run all benchmarks
cargo bench -- "search_full"      # run a specific benchmark group
```

Three benchmark groups:

- **component** — helix/strand/config scoring throughput
- **search** — full search at various sequence sizes (2KB–1MB) and parallelism levels
- **cli** — end-to-end comparison of Rust erspin vs. the original C ERPIN binary

## Project structure

```
src/
  main.rs       CLI entry point (clap derive)
  lib.rs        Library root
  types.rs      Core data structures
  epn.rs        .epn training set parser
  fasta.rs      Streaming FASTA reader
  profile.rs    Log-odds scoring profiles
  region.rs     Region and mask spec parsers
  pattern.rs    Pattern builder (atoms, helices, strands, gap stats)
  scoring.rs    Score table computation
  simd.rs       Optimized scoring (column-major loops, interleaved DP)
  search.rs     Multi-level cascade search with Rayon parallelism
  output.rs     Output formatting (text, JSON, TSV)
  evalue.rs     E-value computation (convolution-based)
  error.rs      Error types
benches/
  search_bench.rs   Criterion benchmarks
erpin5.5.4.serv/    Original C ERPIN v5.5.4 (reference implementation)
```
