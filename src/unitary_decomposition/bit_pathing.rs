use crate::errors::CircuitError;
use crate::unitary_decomposition::utils::gray_code;
use rayon::prelude::*;

pub struct BitPather {
    n: u64,
    encoding: Vec<u64>,
    reverse_lookup: Vec<u64>,
}

impl BitPather {
    pub(crate) fn new(n: u64) -> Self {
        let encoding = gray_code(n);
        let mut reverse_lookup = vec![];
        reverse_lookup.resize(encoding.len(), 0);

        encoding.iter().enumerate().for_each(|(indx, code)| {
            reverse_lookup[*code as usize] = indx as u64;
        });

        Self {
            n,
            encoding,
            reverse_lookup,
        }
    }

    fn valid_path_to_closest(&self, valid_index: usize, start: u64, endpoints: &[u64]) -> Result<Vec<u64>, CircuitError> {
        if endpoints.is_empty() {
            CircuitError::make_str_err("Enpoints must contain a value")
        } else if endpoints.iter().any(|v| -> bool {
            (self.reverse_lookup[(*v) as usize] as usize) < valid_index
        }) {
            CircuitError::make_str_err("All endpoints must be in valid region.")
        } else {
            Ok(self.valid_path_helper(valid_index, vec![(start, vec![])], endpoints))
        }
    }

    /// Node is sorted by `self.reverse_lookup[value]` as is endpoints.
    fn valid_path_helper(&self, valid_index: usize, nodes: Vec<(u64, Vec<u64>)>, endpoints: &[u64]) -> Vec<u64> {
        let contained_values: Vec<_> = nodes.iter().map(|(c,_)| *c).collect();
        let mut new_vals: Vec<(u64, Vec<u64>)> = nodes.into_par_iter().map(|(val, path)| {
            (0 .. self.n).fold(vec![], |mut entries, indx| {
                let mask = 1 << indx;
                let new_val = (val & !mask) | (!val & mask);
                if self.reverse_lookup[new_val as usize] as usize >= valid_index {
                    if !contained_values.contains(&new_val) && !path.contains(&new_val) {
                        let mut new_path = path.clone();
                        new_path.push(val);
                        entries.push((new_val, new_path))
                    }
                }
                entries
            })
        }).flatten().collect();
        new_vals.par_sort_by_key(|(c,_)| *c);
        let result = new_vals.into_iter().try_fold((None, vec![]), |(last_c, mut acc_v), (c, path)| {
            if endpoints.contains(&c) {
                Err((c, path))
            } else {
                Ok(match last_c {
                    Some(last_c) if last_c == c => (Some(last_c), acc_v),
                    _ => {
                        acc_v.push((c, path));
                        (Some(c), acc_v)
                    }
                })
            }
        });
        match result {
            Ok((_, new_vals)) => {
                self.valid_path_helper(valid_index, new_vals, endpoints)
            }
            Err((c, path)) => {
                let mut path = path;
                path.push(c);
                path
            }
        }
    }

    /// Take the list of indices with nonzero values and return the path through them
    /// to the target, returns the bits needed to swap (in the form `1 << index`).
    pub fn path(&self, target: u64, through: &[u64]) -> Result<Vec<(u64, u64)>, CircuitError> {
        if target as usize > self.reverse_lookup.len() {
            CircuitError::make_err(format!(
                "Value to={:?} is greater than encoding length {:?}",
                target,
                self.reverse_lookup.len()
            ))
        } else if through.is_empty() || (through.len() == 1 && through[0] == target){
            Ok(vec![])
        } else {
            let target_index = self.reverse_lookup[target as usize];

            let mut nonzeros = through.to_vec();
            if !nonzeros.contains(&target) {
                nonzeros.push(target);
            }
            nonzeros.par_sort_by_key(|row| self.reverse_lookup[(*row) as usize]);
            nonzeros.retain(|v| self.reverse_lookup[(*v) as usize] >= target_index);
            let mut path = vec![];
            while !nonzeros.is_empty() {
                let last = nonzeros.pop().unwrap();
                if last == target {
                    continue
                }
                let sub_path = self.valid_path_to_closest(target_index as usize, last, &nonzeros)?;
                // Should always have start and end in sub_path.
                sub_path.windows(2).for_each(|entries| {
                    let (from, to) = (entries[0], entries[1]);
                    path.push((from, to))
                });
            }
            Ok(path)
        }
    }
}

fn get_largest_bit_index(n: u64, num: u64) -> Option<u64> {
    if num != 0 {
        for i in (0..n).rev() {
            if num & (1 << i) != 0 {
                return Some(i);
            }
        }
    }
    None
}

/// Return closest item in entries by bitflip, and the distance.
fn find_entry_with_lowest_bitflip(n: u64, target: u64, entries: &[u64], starting_flip: u64) -> Option<(u64, u64)> {
    for entry in entries {
        let num_flips = num_bit_flips(target, *entry);
        if num_flips == starting_flip {
            return Some((*entry, starting_flip));
        }
    }
    if starting_flip < n {
        find_entry_with_lowest_bitflip(n, target, entries, starting_flip + 1)
    } else {
        None
    }
}

fn num_bit_flips(from: u64, to: u64) -> u64 {
    let mut diff = from ^ to;
    if diff == 0 {
        0
    } else {
        let mut count = 0;
        while diff > 0 {
            if diff & 1 == 1 {
                count += 1;
            }
            diff >>= 1;
        }
        count
    }
}


#[cfg(test)]
mod bitpath_tests {
    use super::*;

    macro_rules! numflip_tests {
        ($($name:ident: ($from:expr, $to:expr, $diff:expr),)*) => {
        $(
            #[test]
            fn $name() {
                assert_eq!(num_bit_flips($from, $to), $diff)
            }
        )*
        }
    }

    numflip_tests! {
        numflip_0: (0b00, 0b00, 0),
        numflip_1: (0b01, 0b01, 0),
        numflip_2: (0b10, 0b10, 0),
        numflip_3: (0b00, 0b01, 1),
        numflip_4: (0b01, 0b00, 1),
        numflip_5: (0b00, 0b11, 2),
        numflip_6: (0b100, 0b111, 2),
        numflip_7: (0b110, 0b111, 1),
        numflip_8: (0b111, 0b111, 0),
        numflip_9: (0b00000000, 0b11111111, 8),
    }

    #[test]
    fn testclosest_single_bit_0() {
        let n = 3;
        let entries = [0b000, 0b001, 0b010, 0b011];
        let target = 0b100;

        let result = find_entry_with_lowest_bitflip(n, target, &entries, 1);
        assert_eq!(result, Some((0b000, 1)));
    }

    #[test]
    fn testclosest_single_bit_1() {
        let n = 3;
        let entries = [0b001, 0b010, 0b011];
        let target = 0b100;

        let result = find_entry_with_lowest_bitflip(n, target, &entries, 1);
        assert_eq!(result, Some((0b001, 2)));
    }

    fn test_path(mut value_vec: Vec<bool>, path: Vec<(u64, u64)>, target: usize) {
        for (from, to) in path {
            let (from, to) = (from as usize, to as usize);
            value_vec[to] |= value_vec[from];
            value_vec[from] = false;
        }

        let mut expected = vec![];
        expected.resize(value_vec.len(), false);
        expected[target] = true;

        assert_eq!(value_vec, expected);
    }

    #[test]
    fn path_test_single() -> Result<(), CircuitError> {
        let acc = vec![false, false, false, false, false, false, false, true];
        let pather = BitPather::new(3);
        let target = 0;
        let path = pather.path(target, &[7])?;
        println!("{:?}", path);
        test_path(acc, path, target as usize);
        Ok(())
    }


    #[test]
    fn path_test_multi_anyway() -> Result<(), CircuitError> {
        let acc = vec![false, false, false, true, false, false, false, true];
        let pather = BitPather::new(3);
        let target = 0;
        let path = pather.path(target, &[3, 7])?;
        println!("{:?}", path);
        test_path(acc, path, target as usize);
        Ok(())
    }


    #[test]
    fn path_test_multi_outofway() -> Result<(), CircuitError> {
        let acc = vec![false, false, true, false, false, false, false, true];
        let pather = BitPather::new(3);
        let target = 0;
        let path = pather.path(target, &[2, 7])?;
        println!("{:?}", path);
        test_path(acc, path, target as usize);
        Ok(())
    }


    #[test]
    fn path_test_multi_detour() -> Result<(), CircuitError> {
        let acc = vec![false, false, false, false, false, true, false, true];
        let pather = BitPather::new(3);
        let target = 0;
        let path = pather.path(target, &[5, 7])?;
        println!("{:?}", path);
        test_path(acc, path, target as usize);
        Ok(())
    }
}
