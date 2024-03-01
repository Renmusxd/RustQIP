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

impl<CB: CircuitBuilder + 'static, const N: usize> Circuit<CB, N> {
    /// Init a empty circuit
    pub fn new() -> Self {
        Self {
            func: Rc::new(|_, rs| Ok(rs)),
        }
    }

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
                let f_input = indices.map(|idx| mem::take(&mut out[idx]).unwrap());
                let f_out = f(cb, f_input)?;

                iter::zip(indices, f_out).for_each(|(idx, r)| out[idx] = Some(r));

                Ok(out.map(|r| r.unwrap()))
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

        let [ra, rb] = Circuit::new()
            .apply(gamma, [0, 1])
            .input([ra, rb])
            .run(&mut b)?;

        let r = b.merge_two_registers(ra, rb);
        let (_, _) = b.measure_stochastic(r);

        let (_, _) = b.calculate_state();

        Ok(())
    }
}
