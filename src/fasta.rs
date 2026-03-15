use crate::error::ErspinError;
use crate::types::Sequence;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// A streaming FASTA reader that yields one sequence at a time without loading
/// the entire file into memory.
pub struct FastaReader<R: Read> {
    reader: BufReader<R>,
    /// Buffered header line for the next sequence (already consumed from reader).
    pending_header: Option<String>,
    /// 1-based sequence counter.
    seq_index: usize,
    /// Whether we've reached EOF.
    done: bool,
}

impl FastaReader<File> {
    /// Open a FASTA file for streaming.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, ErspinError> {
        let file = File::open(path).map_err(ErspinError::Io)?;
        Ok(Self::new(file))
    }
}

impl<R: Read> FastaReader<R> {
    pub fn new(reader: R) -> Self {
        Self {
            reader: BufReader::new(reader),
            pending_header: None,
            seq_index: 0,
            done: false,
        }
    }

    /// Read the next sequence from the file.
    /// Returns `Ok(None)` at EOF.
    pub fn next_seq(&mut self) -> Result<Option<Sequence>, ErspinError> {
        if self.done {
            return Ok(None);
        }

        // Find the header line: either from pending_header or by scanning forward.
        let comment = match self.pending_header.take() {
            Some(h) => h,
            None => {
                let mut line = String::new();
                loop {
                    line.clear();
                    let bytes_read = self
                        .reader
                        .read_line(&mut line)
                        .map_err(ErspinError::Io)?;
                    if bytes_read == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                    let trimmed = line.trim();
                    if trimmed.starts_with('>') {
                        break;
                    }
                }
                line.trim()
                    .strip_prefix('>')
                    .unwrap_or("")
                    .trim()
                    .to_string()
            }
        };

        self.seq_index += 1;
        let index = self.seq_index;

        // Read sequence data until the next '>' header or EOF.
        let mut data = Vec::new();
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = self
                .reader
                .read_line(&mut line)
                .map_err(ErspinError::Io)?;
            if bytes_read == 0 {
                self.done = true;
                break;
            }
            let trimmed = line.trim();
            if trimmed.starts_with('>') {
                // This is the header for the next sequence.
                self.pending_header = Some(
                    trimmed
                        .strip_prefix('>')
                        .unwrap_or("")
                        .trim()
                        .to_string(),
                );
                break;
            }
            // Extract only alphabetic characters (skip digits, whitespace, etc.)
            for &b in trimmed.as_bytes() {
                if b.is_ascii_alphabetic() {
                    data.push(preprocess_seq_base(b));
                }
            }
        }

        if data.is_empty() {
            return Err(ErspinError::InvalidFasta(format!(
                "sequence '{}' has no data",
                comment
            )));
        }

        Ok(Some(Sequence {
            comment,
            index,
            data,
        }))
    }

    /// Collect all sequences into a Vec. Use only for small files.
    pub fn collect_all(mut self) -> Result<Vec<Sequence>, ErspinError> {
        let mut seqs = Vec::new();
        while let Some(seq) = self.next_seq()? {
            seqs.push(seq);
        }
        Ok(seqs)
    }
}

/// Preprocess a sequence base: uppercase, U→T, non-ATGC→N.
fn preprocess_seq_base(b: u8) -> u8 {
    match b.to_ascii_uppercase() {
        b'A' => b'A',
        b'T' => b'T',
        b'U' => b'T',
        b'G' => b'G',
        b'C' => b'C',
        _ => b'N',
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn read_simple_fasta() {
        let input = b"\
>seq1 description
ACGTACGT
TTAA
>seq2
GGCC
";
        let reader = FastaReader::new(Cursor::new(input));
        let seqs = reader.collect_all().unwrap();
        assert_eq!(seqs.len(), 2);
        assert_eq!(seqs[0].comment, "seq1 description");
        assert_eq!(seqs[0].data, b"ACGTACGTTTAA");
        assert_eq!(seqs[0].index, 1);
        assert_eq!(seqs[1].comment, "seq2");
        assert_eq!(seqs[1].data, b"GGCC");
        assert_eq!(seqs[1].index, 2);
    }

    #[test]
    fn read_fasta_with_leading_blank_lines() {
        let input = b"\n\n>seq1\nACGT\n";
        let reader = FastaReader::new(Cursor::new(input));
        let seqs = reader.collect_all().unwrap();
        assert_eq!(seqs.len(), 1);
        assert_eq!(seqs[0].data, b"ACGT");
    }

    #[test]
    fn u_converted_to_t() {
        let input = b">rna\nAUGCUU\n";
        let reader = FastaReader::new(Cursor::new(input));
        let seqs = reader.collect_all().unwrap();
        assert_eq!(seqs[0].data, b"ATGCTT");
    }

    #[test]
    fn reverse_complement() {
        let input = b">s\nACGT\n";
        let reader = FastaReader::new(Cursor::new(input));
        let seqs = reader.collect_all().unwrap();
        assert_eq!(seqs[0].reverse_complement(), b"ACGT");
    }

    #[test]
    fn read_test_fasta_file() {
        let reader =
            FastaReader::from_path("tests/data/test.trna.fasta").unwrap();
        let seqs = reader.collect_all().unwrap();
        assert_eq!(seqs.len(), 1);
        // The FASTA file has ~2000 nucleotide characters across multiple lines.
        assert!(seqs[0].data.len() > 1900, "got len {}", seqs[0].data.len());
    }
}
