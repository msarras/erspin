use thiserror::Error;

#[derive(Debug, Error)]
pub enum ErspinError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("parse error in {file} at line {line}: {message}")]
    Parse {
        file: String,
        line: usize,
        message: String,
    },

    #[error("invalid training set: {0}")]
    InvalidTrainingSet(String),

    #[error("invalid region spec '{spec}': {message}")]
    InvalidRegion { spec: String, message: String },

    #[error("invalid mask spec: {0}")]
    InvalidMask(String),

    #[error("invalid FASTA: {0}")]
    InvalidFasta(String),
}
