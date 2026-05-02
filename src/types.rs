/// Nucleotide encoding for single-strand positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Nucleotide {
    A = 0,
    T = 1,
    G = 2,
    C = 3,
    Gap = 4,
    Unknown = 5,
}

impl Nucleotide {
    pub fn from_ascii(b: u8) -> Self {
        match b {
            b'A' | b'a' => Self::A,
            b'T' | b't' | b'U' | b'u' => Self::T,
            b'G' | b'g' => Self::G,
            b'C' | b'c' => Self::C,
            b'-' | b'.' => Self::Gap,
            _ => Self::Unknown,
        }
    }

    pub fn complement(self) -> Self {
        match self {
            Self::A => Self::T,
            Self::T => Self::A,
            Self::G => Self::C,
            Self::C => Self::G,
            other => other,
        }
    }
}

/// Type of a structural element (atom) in the RNA secondary structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomType {
    /// First (5') strand of a helix.
    Helix1,
    /// Second (3') strand of a helix.
    Helix2,
    /// Isolated single strand.
    Strand,
}

/// An atom: a contiguous run of columns with the same model code.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Original element code from the .epn model.
    pub code: i32,
    pub atom_type: AtomType,
    /// Index into the parent pattern's helix or strand array.
    pub element_index: usize,
    /// For Helix1 atoms, the atom index of the paired Helix2.
    pub paired_atom: Option<usize>,
    /// Start column in the training set alignment.
    pub db_begin: usize,
    /// Number of columns in the alignment.
    pub num_columns: usize,
    /// Minimum length in sequence (conserved columns).
    pub min_len: usize,
    /// Maximum length in sequence (all columns).
    pub max_len: usize,
    /// Maximum number of gaps (max_len - min_len).
    pub max_gaps: usize,
}

/// A helix: two complementary base-paired strand segments.
#[derive(Debug, Clone)]
pub struct Helix {
    /// Element code in the model.
    pub id: i32,
    /// Number of base pairs.
    pub helix_len: usize,
    /// Minimum total length in sequence (2*helix_len + min_dist).
    pub min_len: usize,
    /// Maximum total length (2*helix_len + max_dist).
    pub max_len: usize,
    pub max_gaps: usize,
    pub min_dist: usize,
    pub max_dist: usize,
    /// Start column of 5' strand in alignment.
    pub db_begin_5p: usize,
    /// Start column of 3' strand in alignment.
    pub db_begin_3p: usize,
    /// Position of 5' strand start within the pattern (at min gap config).
    pub min_bgn: usize,
    /// Log-odds scoring profile: [dinuc_code][position], `DINUC_CODES` × helix_len.
    pub profile: Vec<Vec<f64>>,
    /// f32 flat-strided copy of `profile` for SIMD scoring.
    /// Layout: `profile_f32[code * helix_len + j]`, length = `DINUC_CODES` * helix_len.
    pub profile_f32: Vec<f32>,
    /// Per-column-max sum: an upper bound on the score this helix can ever
    /// contribute. Used by the DMP scorer for early-termination pruning. Stable
    /// once `profile` is built, so cache it instead of recomputing per hit.
    pub upper_bound: f64,
}

/// A single-stranded region (possibly with gaps).
#[derive(Debug, Clone)]
pub struct Strand {
    /// Element code in the model.
    pub id: i32,
    pub min_len: usize,
    pub max_len: usize,
    pub max_gaps: usize,
    /// Start column in alignment.
    pub db_begin: usize,
    /// Position within the pattern (at min gap config).
    pub min_bgn: usize,
    /// Log-odds scoring profile: [nt_code][position], 6 × max_len.
    pub profile: Vec<Vec<f64>>,
    /// f32 flat-strided copy of `profile` for SIMD scoring.
    /// Layout: `profile_f32[code * max_len + j]`, length = 6 * max_len.
    pub profile_f32: Vec<f32>,
    /// Per-column-max sum: upper bound on this strand's contribution. Same
    /// purpose as `Helix::upper_bound`.
    pub upper_bound: f64,
}

/// A pattern defines the secondary structure layout extracted from the
/// training set region.
#[derive(Debug, Clone)]
pub struct Pattern {
    pub atoms: Vec<Atom>,
    pub helices: Vec<Helix>,
    pub strands: Vec<Strand>,
    /// Total minimum length of the pattern.
    pub min_len: usize,
    /// Total maximum length.
    pub max_len: usize,
    /// First alignment column of the pattern.
    pub db_begin: usize,
}

/// A gap configuration: specific gap assignments for all elements in a mask.
#[derive(Debug, Clone)]
pub struct Config {
    /// Total length of the pattern in this configuration.
    pub len: usize,
    /// Position of each mask strand relative to the mask start.
    pub st_bgn: Vec<usize>,
    /// Gap variant for each mask strand.
    pub st_gaps: Vec<usize>,
    /// Position of each mask helix relative to the mask start.
    pub hx_bgn: Vec<usize>,
    /// Gap variant for each mask helix (distance variation).
    pub hx_gaps: Vec<usize>,
    /// Per-atom gap assignments (indexed by atom index in pattern). Shared via
    /// `Arc` so every Hit produced from this config can refcount-bump instead
    /// of cloning the Vec.
    pub atom_gaps: std::sync::Arc<[usize]>,
}

/// A gap variable: one degree of freedom in the gap configuration. Either a
/// single mask-strand atom (a per-atom gap variable) or a cumulative span of
/// non-mask atoms in the envelope. Cached on `ResolvedMask` so the DMP path
/// can skip re-walking `in_mask` per hit.
#[derive(Debug, Clone)]
pub struct GapVar {
    pub max_gaps: usize,
    /// Atom index range (inclusive start, exclusive end).
    pub atom_range: (usize, usize),
    /// `true` if this variable maps 1:1 to a mask-strand atom (the gap goes
    /// directly to that atom). `false` for non-mask cumulative groups.
    pub is_mask_strand: bool,
}

/// A resolved mask with element indices, precomputed configurations, and
/// derived per-mask scratch (envelope, inverse element maps, gap variables)
/// that is shared between the SMP and DMP paths to avoid recomputing it per
/// hit.
#[derive(Debug, Clone)]
pub struct ResolvedMask {
    /// Indices into pattern.helices.
    pub hx_indices: Vec<usize>,
    /// Indices into pattern.strands.
    pub st_indices: Vec<usize>,
    /// All valid gap configurations.
    pub configs: Vec<Config>,
    /// Cutoff threshold for this level.
    pub threshold: f64,
    /// Minimum start position relative to pattern start.
    pub min_bgn: usize,
    /// Maximum start position relative to pattern start.
    pub max_bgn: usize,
    /// Minimum total length covered.
    pub min_len: usize,
    /// Maximum total length covered.
    pub max_len: usize,
    /// Per-atom membership: `in_mask[atom_idx] == true` iff the atom is part
    /// of this mask. Length = pattern.atoms.len().
    pub in_mask: Vec<bool>,
    /// Inclusive first atom index covered by the mask envelope. `0` if the
    /// mask is empty.
    pub first_atom: usize,
    /// Inclusive last atom index covered by the mask envelope. `0` if empty.
    pub last_atom: usize,
    /// Inverse helix-index map: `hx_inv[pattern_helix_idx] = Some(mask_pos)`
    /// if the helix is in this mask, else `None`. Length = pattern.helices.len().
    pub hx_inv: Vec<Option<usize>>,
    /// Inverse strand-index map: `st_inv[pattern_strand_idx] = Some(mask_pos)`
    /// if the strand is in this mask, else `None`. Length = pattern.strands.len().
    pub st_inv: Vec<Option<usize>>,
    /// Gap variables for this mask, derived from pattern + (hx_indices, st_indices).
    /// Stable across the mask's lifetime; reused by both `generate_configs`
    /// and the per-hit DMP `generate_configs_constrained`.
    pub gap_vars: Vec<GapVar>,
}

/// A parsed training set.
#[derive(Debug, Clone)]
pub struct TrainingSet {
    pub nseq: usize,
    pub alignment_len: usize,
    pub model_digits: usize,
    pub model: Vec<i32>,
    pub sequences: Vec<Vec<u8>>,
    pub comments: Vec<String>,
    pub natom: usize,
    pub nhelix: usize,
    pub nstrand: usize,
    pub atom_list: Vec<i32>,
    pub helix_codes: Vec<i32>,
    pub strand_codes: Vec<i32>,
}

/// A single sequence from a FASTA database.
#[derive(Debug, Clone)]
pub struct Sequence {
    pub comment: String,
    pub index: usize,
    pub data: Vec<u8>,
}

impl Sequence {
    pub fn reverse_complement(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data.len());
        write_reverse_complement(&self.data, &mut out);
        out
    }

    /// Write the reverse-complement of `self.data` into `out` (clearing first),
    /// reusing existing capacity. Used by the search path so the RC buffer can
    /// be pooled across sequences in a thread.
    pub fn reverse_complement_into(&self, out: &mut Vec<u8>) {
        out.clear();
        out.reserve(self.data.len());
        write_reverse_complement(&self.data, out);
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

#[inline]
fn write_reverse_complement(data: &[u8], out: &mut Vec<u8>) {
    out.extend(data.iter().rev().map(|&b| match b {
        b'A' => b'T',
        b'T' => b'A',
        b'G' => b'C',
        b'C' => b'G',
        other => other,
    }));
}

/// Region specification.
#[derive(Debug, Clone)]
pub struct Region {
    pub begin: i32,
    pub end: i32,
}

/// Mask mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskMode {
    /// Use only the specified elements (erspin CLI: bare numbers).
    /// Note: in original ERPIN this was called -umask.
    Mask,
    /// Use all elements except the specified ones (erspin CLI: ! prefix).
    /// Note: in original ERPIN this was called -mask.
    Umask,
    /// Add elements to the previous level's mask.
    Add,
    /// Use all remaining elements.
    NoMask,
}

/// A mask spec from the command line.
#[derive(Debug, Clone)]
pub struct MaskSpec {
    pub mode: MaskMode,
    /// Element IDs (model codes). Empty for NoMask mode.
    pub elements: Vec<usize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MaskProcessing {
    #[default]
    Dynamic,
    Static,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StrandDirection {
    Forward,
    Reverse,
    #[default]
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputStyle {
    #[default]
    Full,
    Compact,
    Quiet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackgroundMode {
    #[default]
    Global,
    Local,
    Uniform,
}

/// A single search hit/detection.
#[derive(Debug, Clone)]
pub struct Hit {
    /// 0-based position in the sequence.
    pub offset: usize,
    /// Length of the match.
    pub length: usize,
    /// Total score.
    pub score: f64,
    /// E-value (if computed).
    pub evalue: Option<f64>,
    /// Direction: forward or reverse complement.
    pub direction: StrandDirection,
    /// Gap configuration index — refers to the configs of the level that
    /// produced this hit.
    pub config_index: usize,
    /// Per-atom gap assignments from the winning config (for DMP propagation).
    /// Shared via `Arc` so producing a Hit from a precomputed config is just a
    /// refcount bump rather than a Vec clone.
    pub atom_gaps: std::sync::Arc<[usize]>,
}
