use crate::error::ErspinError;
use crate::types::{MaskMode, MaskSpec, Region};

/// Parse a region specification string like "-2,2" or "67,31".
///
/// The region defines which portion of the training set alignment to use
/// as the search pattern. Format: `begin,end` where both can be negative.
pub fn parse_region(spec: &str) -> Result<Region, ErspinError> {
    // Handle the case where the spec starts with a negative number.
    // Split on ',' but be careful: "-2,2" should give ["-2", "2"].
    let comma_pos = spec.find(',').ok_or_else(|| ErspinError::InvalidRegion {
        spec: spec.into(),
        message: "expected format: begin,end (e.g., -2,2)".into(),
    })?;

    let begin_str = &spec[..comma_pos];
    let end_str = &spec[comma_pos + 1..];

    let begin: i32 = begin_str
        .parse()
        .map_err(|_| ErspinError::InvalidRegion {
            spec: spec.into(),
            message: format!("invalid begin value: '{}'", begin_str),
        })?;

    let end: i32 = end_str
        .parse()
        .map_err(|_| ErspinError::InvalidRegion {
            spec: spec.into(),
            message: format!("invalid end value: '{}'", end_str),
        })?;

    Ok(Region { begin, end })
}

/// Parse mask specifications from CLI arguments.
///
/// Each level has a mode and optional element indices:
/// - `--levels "6,8 / 2,3 / *"` → three levels
/// - Level separator: `/`
/// - `*` means NoMask (all remaining elements)
/// - Bare numbers mean Mask mode (only those elements)
/// - Prefix `!` means Umask (all except those)
/// - Prefix `+` means Add (add to previous)
pub fn parse_mask_specs(spec: &str) -> Result<Vec<MaskSpec>, ErspinError> {
    let levels: Vec<&str> = spec.split('/').map(str::trim).collect();
    let mut masks = Vec::with_capacity(levels.len());

    for level_str in levels {
        if level_str.is_empty() {
            return Err(ErspinError::InvalidMask(
                "empty level specification".into(),
            ));
        }

        let mask = if level_str == "*" {
            MaskSpec {
                mode: MaskMode::NoMask,
                elements: Vec::new(),
            }
        } else if let Some(rest) = level_str.strip_prefix('!').or_else(|| level_str.strip_prefix("\\!")) {
            MaskSpec {
                mode: MaskMode::Umask,
                elements: parse_element_list(rest)?,
            }
        } else if let Some(rest) = level_str.strip_prefix('+') {
            MaskSpec {
                mode: MaskMode::Add,
                elements: parse_element_list(rest)?,
            }
        } else {
            MaskSpec {
                mode: MaskMode::Mask,
                elements: parse_element_list(level_str)?,
            }
        };

        masks.push(mask);
    }

    Ok(masks)
}

fn parse_element_list(s: &str) -> Result<Vec<usize>, ErspinError> {
    s.split(',')
        .map(|e| {
            e.trim()
                .parse::<usize>()
                .map_err(|_| ErspinError::InvalidMask(format!("invalid element: '{}'", e.trim())))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_negative_region() {
        let r = parse_region("-2,2").unwrap();
        assert_eq!(r.begin, -2);
        assert_eq!(r.end, 2);
    }

    #[test]
    fn parse_positive_region() {
        let r = parse_region("67,31").unwrap();
        assert_eq!(r.begin, 67);
        assert_eq!(r.end, 31);
    }

    #[test]
    fn parse_mask_nomask() {
        let masks = parse_mask_specs("*").unwrap();
        assert_eq!(masks.len(), 1);
        assert_eq!(masks[0].mode, MaskMode::NoMask);
    }

    #[test]
    fn parse_multi_level_masks() {
        let masks = parse_mask_specs("!6,8 / 2,3 / *").unwrap();
        assert_eq!(masks.len(), 3);
        assert_eq!(masks[0].mode, MaskMode::Umask);
        assert_eq!(masks[0].elements, vec![6, 8]);
        assert_eq!(masks[1].mode, MaskMode::Mask);
        assert_eq!(masks[1].elements, vec![2, 3]);
        assert_eq!(masks[2].mode, MaskMode::NoMask);
    }
}
