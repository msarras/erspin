# erspin

A Rust rewrite of [ERPIN](http://rssf.i2bc.paris-saclay.fr/Software/erpin.php) (Easy RNA Profile Identification), a bioinformatics tool that searches for RNA structural motifs in nucleotide sequence databases using profile-based scoring.

Given a training set alignment (`.epn` file) describing an RNA secondary structure, erspin scans FASTA databases for matching motifs using multi-level mask filtering with configurable score thresholds.

## Concepts

**Training sets** (`.epn` files) encode RNA secondary structure as a multiple sequence alignment. Two model digit lines define structural elements: codes appearing twice denote helices (base-paired 5'/3' strands), codes appearing once denote isolated strands.

**Scoring** uses log-odds weight matrices built from the training alignment. Helices use dinucleotide (base-pair) codes (24 values); strands use single-nucleotide codes (6 values). Gapped strands are scored via banded Needleman-Wunsch DP.

**Multi-level search** applies a cascade of masks (subsets of structural elements) with increasing completeness. Each level filters candidates using a score cutoff, so only positions passing level N are evaluated at level N+1. This dramatically reduces computation on large databases.

**Mask specification** syntax:
- Element indices (e.g., `6,8`) — use these elements at this level
- Multiple `--add` flags define levels: first is the initial mask, subsequent add elements cumulatively
- Omit `--add` entirely to use all elements (nomask)

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
erspin search training.epn database.fasta "-2,2" \
  --add 2,4,5,6 --add 3,8,9,10 \
  --cutoff 100% 100% 90%
```

Positional arguments:

| Argument | Description |
|----------|-------------|
| `training_set` | Training set file (`.epn` format) |
| `database` | Database file (FASTA format) |
| `region` | Region specification (e.g., `-2,2` or `1,23`) |

Options:

| Flag | Description | Default |
|------|-------------|---------|
| `--add` | Mask level elements (comma-separated, repeatable). First `--add` is the initial mask, subsequent groups add elements cumulatively. Omit for all elements. | all elements |
| `--cutoff` | Cutoff thresholds per level (space-separated, raw scores or percentages like `100%`) | `100%` |
| `--logzero` | Minimum log-odds value for zero-count positions | `-20.0` |
| `--processing` | `dynamic` or `static` mask processing | `dynamic` |
| `--strand` | `forward`, `reverse`, or `both` | `both` |
| `--output` | `full`, `compact`, or `quiet` | `full` |
| `--background` | `global`, `local`, or `uniform` | `global` |
| `--pseudo-count` | Pseudo-count weight (0.0–1.0) | `0.0002` |
| `--format` | `text`, `json`, or `tsv` | `text` |
| `--cpu` | Number of threads for parallel search | `4` |

### view

Inspect a training set's structure and alignment:

```bash
erspin view training.epn
erspin view training.epn "-2,2"
```

### stats

Show score statistics (min/median/mean/max) for training sequences at each mask level:

```bash
erspin stats training.epn "-2,2" --add 6,8
```

### eval

Estimate E-values for given cutoffs and database size:

```bash
erspin eval training.epn "-2,2" --add 6,8 -m 1.0
```

### configs

Show configuration counts per mask level:

```bash
erspin configs training.epn "-2,2" --add 6,8
```

## Quick start with test data

The repository includes tRNA test data from the original ERPIN distribution:

```bash
cargo build --release

./target/release/erspin search \
  tests/data/trna.typeI.epn \
  tests/data/test.trna.fasta \
  "-2,2" \
  --add 1,2,3,4,5,7,9,10 --add 6,8 \
  --cutoff 100% 100% 90%
```

Expected reference output is in `tests/data/test.txt`.

## Testing

```bash
cargo test           # run all 26 tests
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
- **cli** — end-to-end CLI throughput

## Citation

If you find erspin useful, please cite the original ERPIN paper:

> Gautheret D, Lambert A. (2001) Direct RNA Motif Definition and Identification from Multiple Sequence Alignments using Secondary Structure Profiles. *J Mol Biol.* 313:1003-11. doi:[10.1006/jmbi.2001.5102](https://doi.org/10.1006/jmbi.2001.5102)

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
tests/
  data/             Test data (tRNA training set, FASTA sequences)
```
