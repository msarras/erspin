use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::process::Command;
use std::time::Duration;

use erspin::epn;
use erspin::pattern;
use erspin::profile::Background;
use erspin::region;
use erspin::scoring;
use erspin::search;
use erspin::types::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a random-ish DNA sequence of `len` bytes (deterministic seed).
fn generate_sequence(len: usize, seed: u64) -> Vec<u8> {
    let bases = [b'A', b'T', b'G', b'C'];
    let mut state = seed;
    (0..len)
        .map(|_| {
            // Simple xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            bases[(state & 3) as usize]
        })
        .collect()
}

/// Write a FASTA file with `nseq` sequences of `seq_len` each.
fn write_fasta(path: &str, nseq: usize, seq_len: usize) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).unwrap();
    for i in 0..nseq {
        writeln!(f, ">seq_{} synthetic test sequence", i).unwrap();
        let seq = generate_sequence(seq_len, (i as u64 + 1) * 12345);
        for chunk in seq.chunks(70) {
            f.write_all(chunk).unwrap();
            f.write_all(b"\n").unwrap();
        }
    }
}

/// Load the tRNA pattern and masks (shared setup for benchmarks).
struct TestFixture {
    pat: Pattern,
    masks: Vec<ResolvedMask>,
}

impl TestFixture {
    fn load() -> Self {
        let ts =
            epn::parse_epn("tests/data/trna.typeI.epn").unwrap();
        let reg = Region { begin: -2, end: 2 };
        let bg = Background::default();
        let pat = pattern::build_pattern(&ts, &reg, &bg, 0.0002, -20.0);

        let specs = region::parse_mask_specs("6,8 / !2,3 / *").unwrap();
        let mut masks = search::resolve_masks(&specs, &pat);
        let cutoffs = vec![
            "100%".to_string(),
            "100%".to_string(),
            "90%".to_string(),
        ];
        search::compute_thresholds(&ts, &pat, &mut masks, &cutoffs);

        Self { pat, masks }
    }
}

// ---------------------------------------------------------------------------
// Component benchmarks
// ---------------------------------------------------------------------------

fn bench_helix_score_table(c: &mut Criterion) {
    let fix = TestFixture::load();
    let seq = generate_sequence(10_000, 42);

    let mut group = c.benchmark_group("helix_score_table");
    for (_i, helix) in fix.pat.helices.iter().enumerate() {
        group.throughput(Throughput::Bytes(seq.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("helix", format!("id={} len={}", helix.id, helix.helix_len)),
            &seq,
            |b, seq| {
                b.iter(|| scoring::compute_helix_score_table(black_box(seq), black_box(helix)));
            },
        );
    }
    group.finish();
}

fn bench_strand_score_table(c: &mut Criterion) {
    let fix = TestFixture::load();
    let seq = generate_sequence(10_000, 42);

    let mut group = c.benchmark_group("strand_score_table");
    for strand in &fix.pat.strands {
        group.throughput(Throughput::Bytes(seq.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "strand",
                format!("id={} len={} gaps={}", strand.id, strand.max_len, strand.max_gaps),
            ),
            &seq,
            |b, seq| {
                b.iter(|| scoring::compute_strand_score_table(black_box(seq), black_box(strand)));
            },
        );
    }
    group.finish();
}

fn bench_config_scoring(c: &mut Criterion) {
    let fix = TestFixture::load();
    let seq = generate_sequence(10_000, 42);

    let mut group = c.benchmark_group("config_scoring");
    for (li, mask) in fix.masks.iter().enumerate() {
        let tables = scoring::compute_mask_score_tables(&seq, &fix.pat, mask);
        let scan_len = seq.len().saturating_sub(mask.min_len);

        group.throughput(Throughput::Elements(scan_len as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "level",
                format!(
                    "{} ({}hx+{}st, {}cfg)",
                    li + 1,
                    mask.hx_indices.len(),
                    mask.st_indices.len(),
                    mask.configs.len()
                ),
            ),
            &tables,
            |b, tables| {
                let lookup = scoring::ConfigLookup::build(mask, tables);
                b.iter(|| {
                    for pos in 0..scan_len {
                        black_box(lookup.best_score(pos));
                    }
                });
            },
        );
    }
    group.finish();
}

fn bench_mask_score_tables(c: &mut Criterion) {
    let fix = TestFixture::load();
    let seq = generate_sequence(10_000, 42);

    let mut group = c.benchmark_group("mask_score_tables");
    for (li, mask) in fix.masks.iter().enumerate() {
        group.throughput(Throughput::Bytes(seq.len() as u64));
        group.bench_with_input(
            BenchmarkId::new(
                "level",
                format!(
                    "{} ({}hx+{}st)",
                    li + 1,
                    mask.hx_indices.len(),
                    mask.st_indices.len()
                ),
            ),
            &seq,
            |b, seq| {
                b.iter(|| {
                    scoring::compute_mask_score_tables(black_box(seq), black_box(&fix.pat), black_box(mask))
                });
            },
        );
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Full search benchmarks at various database sizes
// ---------------------------------------------------------------------------

fn bench_search_full(c: &mut Criterion) {
    let fix = TestFixture::load();

    let sizes: &[(usize, &str)] = &[
        (2_000, "2kb"),
        (10_000, "10kb"),
        (100_000, "100kb"),
        (1_000_000, "1Mb"),
    ];

    let mut group = c.benchmark_group("search_full");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &(size, label) in sizes {
        let seq_data = generate_sequence(size, 42);
        let seq = Sequence {
            comment: format!("synthetic_{}", label),
            index: 0,
            data: seq_data,
        };

        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::new("rust", label), &seq, |b, seq| {
            b.iter(|| {
                search::search_full(
                    black_box(&fix.pat),
                    black_box(&fix.masks),
                    black_box(seq),
                    StrandDirection::Both,
                )
            });
        });
    }
    group.finish();
}

fn bench_search_parallel(c: &mut Criterion) {
    let fix = TestFixture::load();

    // Multiple sequences to exercise parallelism.
    let configs: &[(usize, usize, &str)] = &[
        (10, 10_000, "10x10kb"),
        (100, 10_000, "100x10kb"),
        (10, 100_000, "10x100kb"),
    ];

    let mut group = c.benchmark_group("search_parallel");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(10));

    for &(nseq, seq_len, label) in configs {
        let sequences: Vec<Sequence> = (0..nseq)
            .map(|i| Sequence {
                comment: format!("seq_{}", i),
                index: i,
                data: generate_sequence(seq_len, (i as u64 + 1) * 7919),
            })
            .collect();

        let total_bytes: u64 = sequences.iter().map(|s| s.data.len() as u64).sum();
        group.throughput(Throughput::Bytes(total_bytes));
        group.bench_with_input(BenchmarkId::new("rust_parallel", label), &sequences, |b, seqs| {
            b.iter(|| {
                search::search_all_parallel(
                    black_box(&fix.pat),
                    black_box(&fix.masks),
                    black_box(seqs),
                    StrandDirection::Both,
                )
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// CLI benchmark (end-to-end)
// ---------------------------------------------------------------------------

fn bench_cli(c: &mut Criterion) {
    let sizes: &[(usize, usize, &str)] = &[
        (1, 10_000, "10kb"),
        (1, 100_000, "100kb"),
        (1, 1_000_000, "1Mb"),
    ];

    let epn = "tests/data/trna.typeI.epn";

    // Build release binary.
    let build_status = Command::new("cargo")
        .args(["build", "--release"])
        .status()
        .expect("failed to build release binary");
    assert!(build_status.success(), "cargo build --release failed");

    let rust_binary = "target/release/erspin";

    let mut group = c.benchmark_group("cli");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    for &(nseq, seq_len, label) in sizes {
        let fasta_path = format!("/tmp/erspin_bench_{}.fasta", label);
        write_fasta(&fasta_path, nseq, seq_len);

        let total_bytes = (nseq * seq_len) as u64;
        group.throughput(Throughput::Bytes(total_bytes));

        group.bench_with_input(BenchmarkId::new("erspin", label), &fasta_path, |b, fasta| {
            b.iter(|| {
                let output = Command::new(rust_binary)
                    .args([
                        "search",
                        "-t", epn,
                        "-d", fasta,
                        "-r", "-2,2",
                        "-l", "6,8 / !2,3 / *",
                        "-c", "100%,100%,90%",
                        "--output", "quiet",
                    ])
                    .output()
                    .expect("failed to run erspin");
                black_box(output);
            });
        });
    }
    group.finish();
}

// ---------------------------------------------------------------------------
// Criterion groups
// ---------------------------------------------------------------------------

criterion_group!(
    component_benches,
    bench_helix_score_table,
    bench_strand_score_table,
    bench_config_scoring,
    bench_mask_score_tables,
);

criterion_group!(
    search_benches,
    bench_search_full,
    bench_search_parallel,
);

criterion_group!(
    cli_benches,
    bench_cli,
);

criterion_main!(component_benches, search_benches, cli_benches);
