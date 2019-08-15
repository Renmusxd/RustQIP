use crate::errors::CircuitError;
use crate::unitary_decomposition::utils::gray_code;
use rayon::prelude::*;

pub(crate) struct BitPather {
    n: u64,
    reverse_lookup: Vec<u64>,
}

impl BitPather {
    pub(crate) fn new(n: u64) -> Self {
        let encoding = gray_code(n);
        let mut reverse_lookup = vec![];
        reverse_lookup.resize(encoding.len(), 0);

        encoding.into_iter().enumerate().for_each(|(indx, code)| {
            reverse_lookup[code as usize] = indx as u64;
        });

        Self { n, reverse_lookup }
    }

    fn valid_path_to_closest(
        &self,
        valid_index: usize,
        start: u64,
        endpoints: &[u64],
    ) -> Result<Vec<u64>, CircuitError> {
        if endpoints.is_empty() {
            CircuitError::make_str_err("Enpoints must contain a value")
        } else {
            let any_below = endpoints
                .iter()
                .any(|v| (self.reverse_lookup[(*v) as usize] as usize) < valid_index);
            if any_below {
                CircuitError::make_str_err("All endpoints must be in valid region.")
            } else {
                Ok(self.valid_path_helper(valid_index, vec![(start, vec![])], endpoints))
            }
        }
    }

    /// Node is sorted by `self.reverse_lookup[value]` as is endpoints.
    fn valid_path_helper(
        &self,
        valid_index: usize,
        nodes: Vec<(u64, Vec<u64>)>,
        endpoints: &[u64],
    ) -> Vec<u64> {
        let contained_values: Vec<_> = nodes.iter().map(|(c, _)| *c).collect();
        let mut new_vals: Vec<(u64, Vec<u64>)> = nodes
            .into_par_iter()
            .map(|(val, path)| {
                (0..self.n).fold(vec![], |mut entries, indx| {
                    let mask = 1 << indx;
                    let new_val = (val & !mask) | (!val & mask);
                    if self.reverse_lookup[new_val as usize] as usize >= valid_index {
                        let new_val_index = self.reverse_lookup[new_val as usize];
                        let search_result = contained_values
                            .binary_search_by_key(&new_val_index, |c| {
                                self.reverse_lookup[(*c) as usize]
                            });
                        if search_result.is_err() && !path.contains(&new_val) {
                            let mut new_path = path.clone();
                            new_path.push(val);
                            entries.push((new_val, new_path))
                        }
                    }
                    entries
                })
            })
            .flatten()
            .collect();
        new_vals.par_sort_by_key(|(c, _)| self.reverse_lookup[(*c) as usize]);
        let result =
            new_vals
                .into_iter()
                .try_fold((None, vec![]), |(last_c, mut acc_v), (c, path)| {
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
            Ok((_, new_vals)) => self.valid_path_helper(valid_index, new_vals, endpoints),
            Err((c, path)) => {
                let mut path = path;
                path.push(c);
                path
            }
        }
    }

    /// Take the list of indices with nonzero values and return the path through them
    /// to the target, returns the bits needed to swap (in the form `1 << index`).
    pub(crate) fn path(
        &self,
        target: u64,
        through: &[u64],
    ) -> Result<Vec<(u64, u64)>, CircuitError> {
        if target as usize > self.reverse_lookup.len() {
            CircuitError::make_err(format!(
                "Value to={:?} is greater than encoding length {:?}",
                target,
                self.reverse_lookup.len()
            ))
        } else if through.is_empty() || (through.len() == 1 && through[0] == target) {
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
                    continue;
                }
                let sub_path =
                    self.valid_path_to_closest(target_index as usize, last, &nonzeros)?;
                // Should always have start and end in sub_path.
                let step = sub_path[1];
                let step_index = self.reverse_lookup[step as usize];
                let result = nonzeros
                    .binary_search_by_key(&step_index, |c| self.reverse_lookup[(*c) as usize]);
                if let Err(index) = result {
                    nonzeros.insert(index, step);
                }
                path.push((last, step));
            }
            Ok(path)
        }
    }
}

#[cfg(test)]
mod bitpath_tests {
    use super::*;

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
