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