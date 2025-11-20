use std::fmt;
use std::iter;
use std::rc::Rc;

use crate::builder_traits::CircuitBuilder;
use crate::errors::CircuitResult;

type Registers<CB, const N: usize> = [<CB as CircuitBuilder>::Register; N];
type CircuitFunction<CB, const N: usize> =
    dyn Fn(&mut CB, Registers<CB, N>) -> CircuitResult<Registers<CB, N>>;

/// indices for N registers
pub type Idx<const N: usize> = [Vec<(usize, usize)>; N];

/// A subcircuit that you can apply
pub trait AsSubcircuit<CB: CircuitBuilder, const L: usize> {
    /// The innner function of this circuit
    fn circuit_func(self) -> Rc<CircuitFunction<CB, L>>;
}

impl<CB: CircuitBuilder, const L: usize, F> AsSubcircuit<CB, L> for F
where
    F: Fn(&mut CB, Registers<CB, L>) -> CircuitResult<Registers<CB, L>> + 'static,
{
    fn circuit_func(self) -> Rc<CircuitFunction<CB, L>> {
        Rc::new(self)
    }
}

/// Provide indices information when apply a subcircuit
pub trait IndicesInfo<CB: CircuitBuilder, const N: usize, const L: usize>: 'static {
    /// Temporary intermediate type for storing other registers
    type IntermediateRegisters;

    /// Get new registers
    fn get_new_registers(
        &self,
        cb: &mut CB,
        orig_rs: Registers<CB, N>,
    ) -> (Self::IntermediateRegisters, Registers<CB, L>);

    /// Restore original registers
    fn restore_original_registers(
        &self,
        cb: &mut CB,
        itm_rs: Self::IntermediateRegisters,
        sub_rs: Registers<CB, L>,
    ) -> Registers<CB, N>;
}

impl<CB: CircuitBuilder, const N: usize, const L: usize> IndicesInfo<CB, N, L> for [usize; L] {
    type IntermediateRegisters = [Option<CB::Register>; N];

    fn get_new_registers(
        &self,
        _cb: &mut CB,
        orig_rs: Registers<CB, N>,
    ) -> (Self::IntermediateRegisters, Registers<CB, L>) {
        let mut itm_rs = orig_rs.map(Some);
        let sub_rs = self.map(|idx| itm_rs[idx].take().unwrap());
        (itm_rs, sub_rs)
    }

    fn restore_original_registers(
        &self,
        _cb: &mut CB,
        mut itm_rs: Self::IntermediateRegisters,
        sub_rs: Registers<CB, L>,
    ) -> Registers<CB, N> {
        iter::zip(self, sub_rs).for_each(|(&idx, r)| itm_rs[idx] = Some(r));
        itm_rs.map(|r| r.unwrap())
    }
}

impl<CB: CircuitBuilder, const N: usize, const L: usize, MAP> IndicesInfo<CB, N, L> for MAP
where
    MAP: Fn(Idx<N>) -> Idx<L> + 'static,
{
    type IntermediateRegisters = [Vec<Option<CB::Register>>; N];

    fn get_new_registers(
        &self,
        cb: &mut CB,
        orig_rs: Registers<CB, N>,
    ) -> (Self::IntermediateRegisters, Registers<CB, L>) {
        let mut itm_rs = orig_rs.map(|r| {
            let qubits = cb.split_all_register(r);
            qubits.into_iter().map(Some).collect::<Vec<_>>()
        });

        let init_indices: [Vec<(usize, usize)>; N] = itm_rs
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
        let new_indices = self(init_indices);

        let sub_rs = new_indices.map(|qubit_positions| {
            cb.merge_registers(
                qubit_positions
                    .iter()
                    .map(|&(reg_idx, idx)| itm_rs[reg_idx][idx].take().unwrap()),
            )
            .unwrap()
        });
        (itm_rs, sub_rs)
    }

    fn restore_original_registers(
        &self,
        cb: &mut CB,
        mut itm_rs: Self::IntermediateRegisters,
        sub_rs: Registers<CB, L>,
    ) -> Registers<CB, N> {
        let init_indices: [Vec<(usize, usize)>; N] = itm_rs
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
        let new_indices = self(init_indices);

        iter::zip(new_indices, sub_rs.map(|r| cb.split_all_register(r))).for_each(
            |(qubit_positions, out_qubits)| {
                iter::zip(qubit_positions, out_qubits).for_each(|((reg_idx, idx), qubit)| {
                    itm_rs[reg_idx][idx] = Some(qubit);
                });
            },
        );

        itm_rs.map(|qubits| {
            cb.merge_registers(qubits.into_iter().map(|qubit| qubit.unwrap()))
                .unwrap()
        })
    }
}

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

impl<CB: CircuitBuilder, const N: usize> AsSubcircuit<CB, N> for Circuit<CB, N> {
    fn circuit_func(self) -> Rc<CircuitFunction<CB, N>> {
        self.func.clone()
    }
}

impl<CB: CircuitBuilder, const N: usize> AsSubcircuit<CB, N> for &Circuit<CB, N> {
    fn circuit_func(self) -> Rc<CircuitFunction<CB, N>> {
        self.func.clone()
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
    pub fn apply<const L: usize>(
        self,
        subcircuit: impl AsSubcircuit<CB, L>,
        indices: impl IndicesInfo<CB, N, L>,
    ) -> Self {
        let func = self.func.clone();
        let sub_func = subcircuit.circuit_func();
        Self {
            func: Rc::new(move |cb, rs| {
                let out = (*func)(cb, rs)?;
                let (itm, f_input) = indices.get_new_registers(cb, out);
                let f_out = sub_func(cb, f_input)?;

                Ok(indices.restore_original_registers(cb, itm, f_out))
            }),
        }
    }

    /// Apply a function to part of this circuit when flag is true
    pub fn apply_when<const L: usize>(
        self,
        flag: bool,
        subcircuit: impl AsSubcircuit<CB, L>,
        indices: impl IndicesInfo<CB, N, L>,
    ) -> Self {
        if flag {
            self.apply(subcircuit, indices)
        } else {
            self
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

    type CurrentBuilderType = LocalBuilder<f64>;

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
        let mut b = CurrentBuilderType::default();
        let ra = b.try_register(3).unwrap();
        let rb = b.try_register(3).unwrap();

        let gamma_circuit = Circuit::from(gamma);

        let measure_result_0 = 1;

        let [ra, rb] = Circuit::default()
            // Applies gamma to |ra>|rb>
            .apply(gamma, [0, 1])
            // Applies gamma to |rb>|ra> when measure_result_0 = 1
            .apply_when(measure_result_0 == 1, gamma, [1, 0])
            // Applies gamma to |ra[0] ra[1]>|ra[2]>
            .apply(gamma, |[ra, _]: Idx<2>| [ra[0..=1].to_vec(), vec![ra[2]]])
            // Applies gamma to |ra[0] rb[0]>|ra[2]>
            .apply(&gamma_circuit, |[ra, rb]: Idx<2>| {
                [vec![ra[0], rb[0]], vec![ra[2]]]
            })
            // Applies gamma to |ra[0]>|rb[0] ra[2]>
            .apply(gamma_circuit, |[ra, rb]: Idx<2>| {
                [vec![ra[0]], vec![rb[0], ra[2]]]
            })
            // Applies a more complex subcircuit to |ra[1]>|ra[2]>|rb>
            .apply(
                Circuit::default()
                    .apply(gamma, [0, 1])
                    .apply(gamma, [1, 2])
                    .apply(
                        |b: &mut CurrentBuilderType, rs| {
                            let [ra, rb] = rs;
                            let (ra, rb) = b.cnot(ra, rb)?;
                            Ok([ra, rb])
                        },
                        |[_, r2, r3]: Idx<3>| [r2, vec![r3[0]]],
                    ),
                |[ra, rb]: Idx<2>| [vec![ra[1]], vec![ra[2]], rb],
            )
            .input([ra, rb])
            .run(&mut b)?;

        let r = b.merge_two_registers(ra, rb);
        let (_, _) = b.measure_stochastic(r);

        let (_, _) = b.calculate_state();

        Ok(())
    }
}
