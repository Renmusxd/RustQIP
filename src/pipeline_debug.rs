extern crate num;

use std::marker::PhantomData;

use num::complex::Complex;

use crate::measurement_ops::MeasuredCondition;
use crate::pipeline;
use crate::pipeline::{InitialState, QuantumState};
use crate::qubits::Qubit;
use crate::state_ops::{get_index, num_indices, QubitOp};
use crate::types::Precision;

struct PrintPipeline<P: Precision> {
    n: u64,
    phantom: PhantomData<P>,
}

impl<P: Precision> QuantumState<P> for PrintPipeline<P> {
    fn new(n: u64) -> PrintPipeline<P> {
        let tmp: Vec<String> = (0..n).map(|i| i.to_string()).collect();
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0..n).map(|_| "V".to_string()).collect();
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0..n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));

        PrintPipeline {
            n,
            phantom: PhantomData,
        }
    }

    fn new_from_initial_states(
        n: u64,
        _states: &[(Vec<u64>, InitialState<P>)],
    ) -> PrintPipeline<P> {
        PrintPipeline::<P>::new(n)
    }

    fn n(&self) -> u64 {
        self.n
    }

    fn apply_op_with_name(&mut self, name: Option<&str>, op: &QubitOp) {
        match op {
            QubitOp::Control(c_indices, o_indices, _) => {
                let lower = c_indices
                    .iter()
                    .chain(o_indices.iter())
                    .cloned()
                    .min()
                    .unwrap_or(0);
                let upper = c_indices
                    .iter()
                    .chain(o_indices.iter())
                    .cloned()
                    .max()
                    .unwrap_or(self.n);

                for _ in 0..lower {
                    print!("{} ", "|".to_string());
                }
                for i in lower..=upper {
                    let conn = if i == upper { " " } else { "-" };
                    if c_indices.contains(&i) {
                        print!("{}{}", "C".to_string(), conn);
                    } else if o_indices.contains(&i) {
                        print!("{}{}", "O".to_string(), conn);
                    } else {
                        print!("{}{}", "|".to_string(), conn);
                    }
                }
                for _ in upper + 1..self.n {
                    print!("{} ", "|".to_string());
                }
                if let Some(name) = name {
                    print!("\t{}", name);
                }
                println!()
            }
            QubitOp::Swap(a_indices, b_indices) => {
                let lower = a_indices
                    .iter()
                    .chain(b_indices.iter())
                    .cloned()
                    .min()
                    .unwrap_or(0);
                let upper = a_indices
                    .iter()
                    .chain(b_indices.iter())
                    .cloned()
                    .max()
                    .unwrap_or(self.n);

                for _ in 0..lower {
                    print!("{} ", "|".to_string());
                }
                for i in lower..=upper {
                    let conn = if i == upper { " " } else { "-" };
                    if a_indices.contains(&i) {
                        print!("{}{}", "A".to_string(), conn);
                    } else if b_indices.contains(&i) {
                        print!("{}{}", "B".to_string(), conn);
                    } else {
                        print!("{}{}", "|".to_string(), conn);
                    }
                }
                for _ in upper + 1..self.n {
                    print!("{} ", "|".to_string());
                }
                if let Some(name) = name {
                    print!("\t{}", name);
                }
                println!()
            }
            _ => {
                let indices: Vec<u64> = (0..num_indices(op)).map(|i| get_index(op, i)).collect();
                let mut tmp: Vec<String> = vec![];
                for i in 0u64..self.n {
                    if indices.contains(&i) {
                        tmp.push("o".to_string())
                    } else {
                        tmp.push("|".to_string())
                    }
                }
                print!("{}", tmp.join(" "));
                if let Some(name) = name {
                    print!("\t{}", name);
                }
                println!()
            }
        };
        let tmp: Vec<String> = (0..self.n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));
    }

    fn measure(&mut self, indices: &[u64], _: Option<MeasuredCondition<P>>) -> (u64, P) {
        let mut tmp: Vec<String> = vec![];
        for i in 0u64..self.n {
            if indices.contains(&i) {
                tmp.push("M".to_string())
            } else {
                tmp.push("|".to_string())
            }
        }
        println!("{}", tmp.join(" "));
        let tmp: Vec<String> = (0..self.n).map(|_| "|".to_string()).collect();
        println!("{}", tmp.join(" "));
        (0, P::zero())
    }

    fn soft_measure(&self, _: &[u64], _: Option<u64>) -> (u64, P) {
        (0, P::zero())
    }

    fn state_magnitude(&self) -> P {
        P::zero()
    }

    fn stochastic_measure(&self, _indices: &[u64]) -> Vec<P> {
        vec![]
    }

    fn get_state(self, _natural_order: bool) -> Vec<Complex<P>> {
        vec![]
    }
}

pub fn run_debug(q: &Qubit) {
    pipeline::run_with_statebuilder(q, |qs| {
        let n: u64 = qs.iter().map(|q| -> u64 { q.n() }).sum();
        PrintPipeline::<f32>::new(n)
    });
}
