use std::fmt;
use std::iter;
use std::mem;
use std::rc::Rc;

use crate::builder_traits::CircuitBuilder;
use crate::errors::CircuitResult;

/// A circuit described by a function
#[derive(Clone)]
pub struct Circuit<CB: CircuitBuilder, const N: usize> {
    func: Rc<dyn Fn(&mut CB, [CB::Register; N]) -> CircuitResult<[CB::Register; N]>>,
}

impl<CB: CircuitBuilder, const N: usize> std::fmt::Debug for Circuit<CB, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Circuit").field("N", &N).finish()
    }
}

impl<CB: CircuitBuilder, const N: usize> Default for Circuit<CB, N> {
    /// Init a empty circuit
    fn default() -> Self {
        Self {
            func: Rc::new(|_, rs| Ok(rs)),
        }
    }
}

impl<CB: CircuitBuilder + 'static, const N: usize> Circuit<CB, N> {
    /// Apply a function to part of this circuit
    pub fn apply<F, const L: usize>(self, f: F, indices: [usize; L]) -> Self
    where
        F: Fn(&mut CB, [CB::Register; L]) -> CircuitResult<[CB::Register; L]> + 'static,
    {
        let func = self.func.clone();
        Self {
            func: Rc::new(move |cb, rs| {
                let out = (*func)(cb, rs)?;
                let mut out = out.map(Some);
                let f_input = indices.map(|idx| out[idx].take().unwrap());
                let f_out = f(cb, f_input)?;

                iter::zip(indices, f_out).for_each(|(idx, r)| out[idx] = Some(r));

                Ok(out.map(|r| r.unwrap()))
            }),
        }
    }

    /// Apply a sub circuit for specific qubits under some new indices combine
    pub fn under_new_indices<const L: usize>(
        self,
        new_indices: [&[(usize, usize)]; L],
        sub_circuit: Circuit<CB, L>,
    ) -> Self {
        let func = self.func.clone();
        let new_indices = new_indices.map(|idx_pairs| idx_pairs.to_vec());

        Self {
            func: Rc::new(move |cb, rs| {
                let out = (*func)(cb, rs)?;
                let mut out = out.map(RegisterRepr::Origin);

                // combine to new  registers
                let f_input = new_indices.clone().map(|idx_pairs| {
                    let rs = idx_pairs
                        .iter()
                        .map(|&(reg_idx, idx)| {
                            if matches!(out[reg_idx], RegisterRepr::Origin(_)) {
                                let r =
                                    mem::replace(out.get_mut(reg_idx).unwrap(), RegisterRepr::Tmp);
                                let RegisterRepr::Origin(r) = r else {
                                    unreachable!()
                                };
                                let qubits = cb.split_all_register(r);
                                out[reg_idx] =
                                    RegisterRepr::Splited(qubits.into_iter().map(Some).collect());
                            }

                            let RegisterRepr::Splited(rs) = out.get_mut(reg_idx).unwrap() else {
                                unreachable!()
                            };
                            rs[idx].take().unwrap()
                        })
                        .collect::<Vec<_>>();

                    cb.merge_registers(rs).unwrap()
                });

                let f_output = (*sub_circuit.func)(cb, f_input)?;
                let f_output_qubits = f_output.map(|r| cb.split_all_register(r));

                // restore
                iter::zip(new_indices.clone(), f_output_qubits).for_each(|(idx_pairs, qubits)| {
                    iter::zip(idx_pairs, qubits).for_each(|((reg_idx, idx), qubit)| {
                        let RegisterRepr::Splited(rs) = out.get_mut(reg_idx).unwrap() else {
                            unreachable!()
                        };
                        rs[idx] = Some(qubit);
                    });
                });

                Ok(out.map(|rr| rr.into_origin(cb)))
            }),
        }
    }

    /// Set input for circuit
    pub fn input(self, input: [CB::Register; N]) -> CircuitWithInput<CB, N> {
        CircuitWithInput {
            circuit: self,
            input,
        }
    }
}

enum RegisterRepr<CB: CircuitBuilder> {
    Origin(CB::Register),
    Splited(Vec<Option<CB::Register>>),
    Tmp,
}

impl<CB: CircuitBuilder> RegisterRepr<CB> {
    fn into_origin(self, cb: &mut CB) -> CB::Register {
        match self {
            Self::Origin(r) => r,
            Self::Splited(qubits) => cb
                .merge_registers(qubits.into_iter().map(|r| r.unwrap()))
                .unwrap(),
            Self::Tmp => unreachable!(),
        }
    }
}

/// Circuit with input
#[derive(Debug)]
pub struct CircuitWithInput<CB: CircuitBuilder, const N: usize> {
    circuit: Circuit<CB, N>,
    input: [CB::Register; N],
}

impl<CB: CircuitBuilder, const N: usize> CircuitWithInput<CB, N> {
    /// Run the circuit
    pub fn run(self, cb: &mut CB) -> CircuitResult<[CB::Register; N]> {
        (*self.circuit.func)(cb, self.input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    fn gamma<B>(b: &mut B, rs: [B::Register; 2]) -> CircuitResult<[B::Register; 2]>
    where
        B: AdvancedCircuitBuilder<f64>,
    {
        let [ra, rb] = rs;
        let (ra, rb) = b.toffoli(ra, rb)?;
        let (rb, ra) = b.toffoli(rb, ra)?;
        Ok([ra, rb])
    }

    #[test]
    fn test_chain_circuit() -> CircuitResult<()> {
        let mut b = LocalBuilder::default();
        let ra = b.try_register(3).unwrap();
        let rb = b.try_register(3).unwrap();

        let [ra, rb] = Circuit::default()
            // Applies gamma to |ra>|rb>
            .apply(gamma, [0, 1])
            // Applies gamma to |ra[0] ra[1]>|ra[2]>
            .under_new_indices(
                [&[(0, 0), (0, 1)], &[(0, 2)]],
                Circuit::default().apply(gamma, [0, 1]),
            )
            .input([ra, rb])
            .run(&mut b)?;

        let r = b.merge_two_registers(ra, rb);
        let (_, _) = b.measure_stochastic(r);

        let (_, _) = b.calculate_state();

        Ok(())
    }
}
