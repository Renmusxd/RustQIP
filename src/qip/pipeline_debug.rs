use crate::pipeline;
use crate::pipeline::{QuantumState, InitialState};
use crate::state_ops::{QubitOp, get_index, num_indices};
use crate::qubits::Qubit;
use std::cmp::{min, max};

struct PrintPipeline {
    n: u64
}



impl QuantumState for PrintPipeline {
    fn new(n: u64) -> Self {
        let tmp: Vec<String> = (0 .. n).map(|i| i.to_string()).collect();
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0 .. n).map(|_| "V".to_string()).collect();
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0 .. n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));

        PrintPipeline {
            n
        }
    }

    fn new_from_initial_states(n: u64, states: &[(Vec<u64>, InitialState)]) -> Self {
        Self::new(n)
    }

    fn apply_op(&mut self, op: &QubitOp) {
         match op {
             QubitOp::ControlOp(c_indices, o_indices, _) => {
                let lower = c_indices.iter().chain(o_indices.iter()).cloned().min().unwrap_or(0);
                let upper = c_indices.iter().chain(o_indices.iter()).cloned().max().unwrap_or(self.n);

                for i in 0 .. lower {
                    print!("{} ", "|".to_string());
                }
                for i in lower .. upper+1 {
                    let conn = if i == upper {
                        " "
                    } else {
                        "-"
                    };
                    if c_indices.contains(&i) {
                        print!("{}{}", "C".to_string(), conn);
                    } else if o_indices.contains(&i) {
                        print!("{}{}", "O".to_string(), conn);
                    } else {
                        print!("{}{}", "|".to_string(), conn);
                    }
                }
                for i in upper+1 .. self.n {
                    print!("{} ", "|".to_string());
                }
                println!()
             },
             QubitOp::SwapOp(a_indices, b_indices) => {
                 let lower = a_indices.iter().chain(b_indices.iter()).cloned().min().unwrap_or(0);
                 let upper = a_indices.iter().chain(b_indices.iter()).cloned().max().unwrap_or(self.n);

                 for i in 0 .. lower {
                     print!("{} ", "|".to_string());
                 }
                 for i in lower .. upper+1 {
                     let conn = if i == upper {
                         " "
                     } else {
                         "-"
                     };
                     if a_indices.contains(&i) {
                         print!("{}{}", "A".to_string(), conn);
                     } else if b_indices.contains(&i) {
                         print!("{}{}", "B".to_string(), conn);
                     } else {
                         print!("{}{}", "|".to_string(), conn);
                     }
                 }
                 for i in upper+1 .. self.n {
                     print!("{} ", "|".to_string());
                 }
                 println!()
             },
             _ => {
                let mut indices: Vec<u64> = (0 .. num_indices(op)).map(|i| get_index(op, i)).collect();
                let mut tmp: Vec<String> = vec![];
                for i in 0u64 .. self.n {
                    if indices.contains(&i) {
                        tmp.push("o".to_string())
                    } else {
                        tmp.push("|".to_string())
                    }
                }
                println!("{}", tmp.join(" "));
            }
        };
        let tmp: Vec<String> = (0 .. self.n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));
    }

    fn measure(&mut self, indices: &Vec<u64>) -> (u64, f64) {
        let mut tmp: Vec<String> = vec![];
        for i in 0u64 .. self.n {
            if indices.contains(&i) {
                tmp.push("M".to_string())
            } else {
                tmp.push("|".to_string())
            }
        }
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0 .. self.n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));
        (0, 0.0)
    }
}

pub fn run_debug(q: &Qubit) {
    pipeline::run_with_statebuilder(q, |qs| {
        let n: u64 = qs.iter().map(|q| -> u64 {
            q.n()
        }).sum();
        PrintPipeline::new(n)
    });
}