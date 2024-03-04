use std::fmt;
use std::iter;
use std::rc::Rc;

use crate::builder_traits::CircuitBuilder;
use crate::errors::CircuitResult;

type Registers<CB, const N: usize> = [<CB as CircuitBuilder>::Register; N];
type CircuitFunction<CB, const N: usize> =
    dyn Fn(&mut CB, Registers<CB, N>) -> CircuitResult<Registers<CB, N>>;

/// A circuit described by a function
#[derive(Clone)]
pub struct Circuit<CB: CircuitBuilder, const N: usize> {
    func: Rc<CircuitFunction<CB, N>>,
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
    /// From a function
    pub fn from<F>(f: F) -> Self
    where
        F: Fn(&mut CB, Registers<CB, N>) -> CircuitResult<Registers<CB, N>> + 'static,
    {
        Self::default().apply(f, core::array::from_fn(|i| i))
    }

    /// Apply a function to part of this circuit
    pub fn apply<F, const L: usize>(self, f: F, indices: [usize; L]) -> Self
    where
        F: Fn(&mut CB, Registers<CB, L>) -> CircuitResult<Registers<CB, L>> + 'static,
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
    pub fn apply_subcircuit<MAP, const L: usize>(
        self,
        indices_map: MAP,
        sub_circuit: &Circuit<CB, L>,
    ) -> Self
    where
        MAP: Fn([Vec<(usize, usize)>; N]) -> [Vec<(usize, usize)>; L] + 'static,
    {
        let func = self.func.clone();
        let sub_func = sub_circuit.func.clone();

        Self {
            func: Rc::new(move |cb, rs| {
                let out = (*func)(cb, rs)?;

                //split
                let mut out = out.map(|r| {
                    let qubits = cb.split_all_register(r);
                    qubits.into_iter().map(Some).collect::<Vec<_>>()
                });

                // combine to new registers
                let init_indices: [Vec<(usize, usize)>; N] = out
                    .iter()
                    .enumerate()
                    .map(|(reg_idx, qubits)| {
                        (0..qubits.len())
                            .map(|idx| (reg_idx, idx))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap();
                let new_indices = indices_map(init_indices);

                let f_input = new_indices.clone().map(|qubit_positions| {
                    cb.merge_registers(
                        qubit_positions
                            .iter()
                            .map(|&(reg_idx, idx)| out[reg_idx][idx].take().unwrap()),
                    )
                    .unwrap()
                });

                let f_output = (*sub_func)(cb, f_input)?;
                let f_output_qubits = f_output.map(|r| cb.split_all_register(r));

                // restore
                iter::zip(new_indices, f_output_qubits).for_each(
                    |(qubit_positions, out_qubits)| {
                        iter::zip(qubit_positions, out_qubits).for_each(
                            |((reg_idx, idx), qubit)| {
                                out[reg_idx][idx] = Some(qubit);
                            },
                        );
                    },
                );

                Ok(out.map(|qubits| {
                    cb.merge_registers(qubits.into_iter().map(|qubit| qubit.unwrap()))
                        .unwrap()
                }))
            }),
        }
    }

    /// Set input for circuit
    pub fn input(self, input: Registers<CB, N>) -> CircuitWithInput<CB, N> {
        CircuitWithInput {
            circuit: self,
            input,
        }
    }
}

/// Circuit with input
#[derive(Debug)]
pub struct CircuitWithInput<CB: CircuitBuilder, const N: usize> {
    circuit: Circuit<CB, N>,
    input: Registers<CB, N>,
}

impl<CB: CircuitBuilder, const N: usize> CircuitWithInput<CB, N> {
    /// Run the circuit
    pub fn run(self, cb: &mut CB) -> CircuitResult<Registers<CB, N>> {
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

        let gamma_circuit = Circuit::from(gamma);

        let [ra, rb] = Circuit::default()
            // Applies gamma to |ra>|rb>
            .apply(gamma, [0, 1])
            // Applies gamma to |ra[0] ra[1]>|ra[2]>
            .apply_subcircuit(|[ra, _]| [ra[0..=1].to_vec(), vec![ra[2]]], &gamma_circuit)
            // Applies gamma to |ra[0] rb[0]>|ra[2]>
            .apply_subcircuit(|[ra, rb]| [vec![ra[0], rb[0]], vec![ra[2]]], &gamma_circuit)
            // Applies gamma to |ra[0]>|rb[0] ra[2]>
            .apply_subcircuit(|[ra, rb]| [vec![ra[0]], vec![rb[0], ra[2]]], &gamma_circuit)
            .input([ra, rb])
            .run(&mut b)?;

        let r = b.merge_two_registers(ra, rb);
        let (_, _) = b.measure_stochastic(r);

        let (_, _) = b.calculate_state();

        Ok(())
    }
}
