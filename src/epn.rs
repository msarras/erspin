use crate::error::ErspinError;
use crate::types::TrainingSet;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

/// Parse a .epn training set file.
///
/// Format:
/// ```text
/// >header comment (ignored)
/// 00000000111100001111    <- digit line 1
/// 12345678123412341234    <- digit line 2 (optional, for codes >= 10)
/// >sequence 1 header
/// -GGGCGAA...ACCA
/// >sequence 2 header
/// -GGGCTCG...CCA-
/// ...
/// ```
///
/// The digit lines encode the secondary structure model. Each column's digits
/// are read vertically and concatenated to form an integer element code.
/// Codes appearing twice identify helices; codes appearing once identify
/// isolated strands.
pub fn parse_epn(path: impl AsRef<Path>) -> Result<TrainingSet, ErspinError> {
    let path = path.as_ref();
    let content = fs::read_to_string(path).map_err(ErspinError::Io)?;
    let filename = path.display().to_string();

    parse_epn_str(&content, &filename)
}

pub fn parse_epn_str(content: &str, filename: &str) -> Result<TrainingSet, ErspinError> {
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err(ErspinError::InvalidTrainingSet(
            "empty file".into(),
        ));
    }

    // First line must be a header (starts with '>').
    if !lines[0].starts_with('>') {
        return Err(ErspinError::Parse {
            file: filename.into(),
            line: 1,
            message: "expected '>' header line".into(),
        });
    }

    // Determine how many digit lines follow the header.
    // Digit lines are lines that consist entirely of digits (0-9).
    let mut digit_lines: Vec<&str> = Vec::new();
    let mut cursor = 1;
    while cursor < lines.len() && is_digit_line(lines[cursor]) {
        digit_lines.push(lines[cursor]);
        cursor += 1;
    }

    if digit_lines.is_empty() {
        return Err(ErspinError::InvalidTrainingSet(
            "no model digit lines found after header".into(),
        ));
    }

    let alignment_len = digit_lines[0].len();
    for (i, dl) in digit_lines.iter().enumerate() {
        if dl.len() != alignment_len {
            return Err(ErspinError::Parse {
                file: filename.into(),
                line: 2 + i,
                message: format!(
                    "digit line length {} differs from first digit line length {}",
                    dl.len(),
                    alignment_len
                ),
            });
        }
    }

    // Decode the model: read columns vertically, concatenate digits, parse as integer.
    let model_digits = digit_lines.len();
    let digit_bytes: Vec<&[u8]> = digit_lines.iter().map(|l| l.as_bytes()).collect();
    let mut model = Vec::with_capacity(alignment_len);

    for col in 0..alignment_len {
        let mut code_str = String::with_capacity(model_digits);
        for row in &digit_bytes {
            code_str.push(row[col] as char);
        }
        let code: i32 = code_str.parse().map_err(|_| ErspinError::Parse {
            file: filename.into(),
            line: 2,
            message: format!("invalid model code '{}' at column {}", code_str, col),
        })?;
        model.push(code);
    }

    // Parse sequences: alternating > header lines and sequence data lines.
    let mut sequences: Vec<Vec<u8>> = Vec::new();
    let mut comments: Vec<String> = Vec::new();

    while cursor < lines.len() {
        let line = lines[cursor];
        if line.starts_with('>') {
            comments.push(line[1..].trim().to_string());
            cursor += 1;

            // Read the sequence line(s) until next header or EOF.
            let mut seq_data = Vec::with_capacity(alignment_len);
            while cursor < lines.len() && !lines[cursor].starts_with('>') {
                for &b in lines[cursor].as_bytes() {
                    if b.is_ascii_alphabetic() || b == b'-' {
                        seq_data.push(preprocess_base(b));
                    }
                }
                cursor += 1;
            }

            if seq_data.len() != alignment_len {
                return Err(ErspinError::Parse {
                    file: filename.into(),
                    line: cursor,
                    message: format!(
                        "sequence '{}' length {} differs from alignment length {}",
                        comments.last().unwrap_or(&String::new()),
                        seq_data.len(),
                        alignment_len,
                    ),
                });
            }

            sequences.push(seq_data);
        } else {
            cursor += 1;
        }
    }

    let nseq = sequences.len();
    if nseq == 0 {
        return Err(ErspinError::InvalidTrainingSet(
            "no sequences found".into(),
        ));
    }

    // Identify structural elements from the model.
    // Build ordered list of unique element codes (left-to-right first occurrence).
    let mut atom_list: Vec<i32> = Vec::new();
    let mut code_positions: BTreeMap<i32, Vec<usize>> = BTreeMap::new();

    for (col, &code) in model.iter().enumerate() {
        code_positions.entry(code).or_default().push(col);
        if !atom_list.contains(&code) {
            atom_list.push(code);
        }
    }

    // Codes appearing exactly twice are helices; once are strands.
    // Count occurrences by contiguous runs, not individual columns.
    let mut run_counts: BTreeMap<i32, usize> = BTreeMap::new();
    let mut prev_code: Option<i32> = None;
    for &code in &model {
        if Some(code) != prev_code {
            *run_counts.entry(code).or_insert(0) += 1;
            prev_code = Some(code);
        }
    }

    let helix_codes: Vec<i32> = atom_list
        .iter()
        .copied()
        .filter(|c| run_counts.get(c) == Some(&2))
        .collect();

    let strand_codes: Vec<i32> = atom_list
        .iter()
        .copied()
        .filter(|c| run_counts.get(c) == Some(&1))
        .collect();

    let nhelix = helix_codes.len();
    let nstrand = strand_codes.len();
    let natom = nhelix * 2 + nstrand;

    Ok(TrainingSet {
        nseq,
        alignment_len,
        model_digits,
        model,
        sequences,
        comments,
        natom,
        nhelix,
        nstrand,
        atom_list,
        helix_codes,
        strand_codes,
    })
}

/// Check whether a line consists entirely of ASCII digits.
fn is_digit_line(line: &str) -> bool {
    !line.is_empty() && line.bytes().all(|b| b.is_ascii_digit())
}

/// Preprocess a base: uppercase, U→T. Non-alphabetic passes through.
fn preprocess_base(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'U' => b'T',
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_epn() {
        let content = "\
>test structure
01001
24024
>seq1
-ACGT
>seq2
-TGCA
";
        let ts = parse_epn_str(content, "test.epn").unwrap();
        assert_eq!(ts.alignment_len, 5);
        assert_eq!(ts.nseq, 2);
        assert_eq!(ts.model_digits, 2);
        // Model codes: column 0 = "01" = 1, column 1 = "24" = 24, etc.
        assert_eq!(ts.model, vec![2, 14, 0, 2, 14]);
        // Code 2 appears twice (col 0 and col 3) => helix
        // Code 14 appears twice (col 1 and col 4) => helix
        // Code 0 appears once (col 2) => strand
        assert_eq!(ts.nhelix, 2);
        assert_eq!(ts.nstrand, 1);
    }

    #[test]
    fn parse_trna_epn() {
        let ts = parse_epn("erpin5.5.4.serv/start.test/trna.typeI.epn").unwrap();
        assert_eq!(ts.alignment_len, 80);
        assert!(ts.nseq > 100);
        // tRNA type I should have 4 helices and several strands
        assert!(ts.nhelix >= 4, "expected >= 4 helices, got {}", ts.nhelix);
        assert!(ts.nstrand >= 4, "expected >= 4 strands, got {}", ts.nstrand);
    }

    #[test]
    fn preprocess_converts_u_to_t() {
        assert_eq!(preprocess_base(b'u'), b'T');
        assert_eq!(preprocess_base(b'U'), b'T');
        assert_eq!(preprocess_base(b'a'), b'A');
    }
}
