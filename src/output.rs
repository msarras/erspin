use anyhow::Result;
use serde::Serialize;

use crate::types::*;

/// A serializable hit record for JSON/TSV output.
#[derive(Serialize)]
struct HitRecord {
    sequence: String,
    direction: String,
    start: usize,
    end: usize,
    length: usize,
    score: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    evalue: Option<f64>,
}

/// Write search results in the specified format.
pub fn write_results(
    results: &[(usize, Vec<Hit>)],
    sequences: &[Sequence],
    format: &str,
    style: OutputStyle,
) -> Result<()> {
    match format {
        "json" => write_json(results, sequences),
        "tsv" => write_tsv(results, sequences, style),
        _ => write_text(results, sequences, style),
    }
}

fn make_records<'a>(
    results: &'a [(usize, Vec<Hit>)],
    sequences: &'a [Sequence],
) -> Vec<HitRecord> {
    let mut records = Vec::new();
    for (seq_idx, hits) in results {
        let seq = &sequences[*seq_idx];
        for hit in hits {
            records.push(HitRecord {
                sequence: seq.comment.clone(),
                direction: match hit.direction {
                    StrandDirection::Forward => "FW".into(),
                    StrandDirection::Reverse => "RC".into(),
                    StrandDirection::Both => "??".into(),
                },
                start: hit.offset + 1,
                end: hit.offset + hit.length,
                length: hit.length,
                score: (hit.score * 100.0).round() / 100.0,
                evalue: hit.evalue,
            });
        }
    }
    records
}

fn write_json(results: &[(usize, Vec<Hit>)], sequences: &[Sequence]) -> Result<()> {
    let records = make_records(results, sequences);
    println!("{}", serde_json::to_string_pretty(&records)?);
    Ok(())
}

fn write_tsv(
    results: &[(usize, Vec<Hit>)],
    sequences: &[Sequence],
    style: OutputStyle,
) -> Result<()> {
    if !matches!(style, OutputStyle::Quiet) {
        println!("sequence\tdirection\tstart\tend\tlength\tscore\tevalue");
    }
    let records = make_records(results, sequences);
    for r in &records {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{:.2}\t{}",
            r.sequence,
            r.direction,
            r.start,
            r.end,
            r.length,
            r.score,
            r.evalue.map_or("-".into(), |e| format!("{:.2e}", e)),
        );
    }
    Ok(())
}

fn write_text(
    results: &[(usize, Vec<Hit>)],
    sequences: &[Sequence],
    style: OutputStyle,
) -> Result<()> {
    if matches!(style, OutputStyle::Quiet) {
        let total: usize = results.iter().map(|(_, h)| h.len()).sum();
        println!("{}", total);
        return Ok(());
    }

    for (seq_idx, hits) in results {
        let seq = &sequences[*seq_idx];
        if matches!(style, OutputStyle::Full) {
            println!(">{}", seq.comment);
        }
        for hit in hits {
            let dir_str = match hit.direction {
                StrandDirection::Forward => "FW",
                StrandDirection::Reverse => "RC",
                StrandDirection::Both => "??",
            };
            let pos1 = hit.offset + 1;
            let end1 = hit.offset + hit.length;
            let evalue_str = hit
                .evalue
                .map_or(String::new(), |e| format!("  {:.2e}", e));

            if matches!(style, OutputStyle::Compact) {
                println!(
                    "{}\t{}\t{}\t{}\t{:.2}{}",
                    seq.comment, dir_str, pos1, end1, hit.score, evalue_str
                );
            } else {
                println!(
                    "{} {:>7}..{:<7}  {:.2}{}",
                    dir_str, pos1, end1, hit.score, evalue_str
                );
            }
        }
    }

    Ok(())
}
