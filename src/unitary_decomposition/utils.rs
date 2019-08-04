fn gray_code(n: u64) -> Vec<u64> {
    if n == 0 {
        vec![]
    } else if n == 1 {
        vec![0, 1]
    } else {
        let subgray = gray_code(n - 1);
        let reflected: Vec<_> = subgray.clone().into_iter().rev().collect();
        let lhs = subgray;
        let rhs: Vec<_> = reflected.into_iter().map(|x| x | (1 << (n-1))).collect();
        lhs.into_iter().chain(rhs.into_iter()).collect()
    }
}


// Need a good algorithm for consolidating entries in sparse unitary matrices other than iterating
// through all the gray codes, this is basically a Steiner tree on a graph where the graph is the
// vertices of a n-dimensional hypercube, it just so happens a paper was written on this:
// https://www.researchgate.net/publication/220617458_Near_Optimal_Bounds_for_Steiner_Trees_in_the_Hypercube


#[cfg(test)]
mod unitary_decomp_tests {
    use super::*;

    fn test_single_bit_set(mut n: u64) -> bool {
        while n > 0 {
            let lower_bit = (n & 1) == 1;
            n >>= 1;

            if lower_bit {
                if n == 0 {
                    return true;
                }
                break;
            }
        }
        false
    }

    #[test]
    fn test_graycodes() {
        for n in 1 .. 10 {
            let codes = gray_code(n);
            for (i, code) in codes[..codes.len()-1].iter().enumerate() {
                let next_code = codes[i+1];
                let diff = *code ^ next_code;
                assert!(test_single_bit_set(diff))
            }
        }
    }
}