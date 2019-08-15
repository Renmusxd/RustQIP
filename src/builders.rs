use crate::errors::CircuitError;
use crate::pipeline::*;
use crate::qubits::*;
use crate::state_ops::*;
use crate::Complex;
use num::Zero;
use std::fmt;

/// A function which takes a builder, a Register, and a set of measured values, and constructs a
/// circuit, outputting the resulting Register.
type SingleRegisterSideChannelFn =
    dyn Fn(&mut dyn UnitaryBuilder, Register, &[u64]) -> Result<Register, CircuitError>;

/// A function which takes a builder, a vec of Register, and a set of measured values, and constructs a
/// circuit, outputting the resulting Registers.
type SideChannelFn =
    dyn Fn(&mut dyn UnitaryBuilder, Vec<Register>, &[u64]) -> Result<Vec<Register>, CircuitError>;

/// A function which takes a builder, a Register with possibly extra indices, and a set of measured
/// values, and constructs a circuit, outputting the resulting Register.
type SideChannelHelperFn =
    dyn Fn(&mut dyn UnitaryBuilder, Register, &[u64]) -> Result<Vec<Register>, CircuitError>;

/// A function to build rows of a sparse matrix. Takes a row and outputs columns and entries.
type SparseBuilderFn = dyn Fn(u64) -> Vec<(u64, Complex<f64>)>;

/// A builder which support unitary operations
pub trait UnitaryBuilder {
    // Things like X, Y, Z, NOT, H, SWAP, ... go here

    /// Build a builder which uses `r` as a condition.
    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder;

    /// Build a generic matrix op, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^(2n)).
    fn mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<Register, CircuitError>;

    /// Build a matrix op from real numbers, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^(2n)).
    fn real_mat(&mut self, name: &str, r: Register, mat: &[f64]) -> Result<Register, CircuitError> {
        self.mat(name, r, from_reals(mat))
    }

    /// Build a sparse matrix op, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^n).
    fn sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Register, CircuitError>;

    /// Build a sparse matrix op from `f`, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^n).
    fn sparse_mat_from_fn(
        &mut self,
        name: &str,
        r: Register,
        f: Box<SparseBuilderFn>,
        natural_order: bool,
    ) -> Result<Register, CircuitError> {
        let n = r.indices.len();
        let mat = make_sparse_matrix_from_function(n, f, natural_order);
        self.sparse_mat(name, r, mat, false)
    }

    /// Build a sparse matrix op from real numbers, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^n).
    fn real_sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: &[Vec<(u64, f64)>],
        natural_order: bool,
    ) -> Result<Register, CircuitError> {
        let mat = mat
            .iter()
            .map(|v| {
                v.iter()
                    .cloned()
                    .map(|(c, r)| (c, Complex { re: r, im: 0.0 }))
                    .collect()
            })
            .collect();
        self.sparse_mat(name, r, mat, natural_order)
    }

    /// Apply NOT to `r`, if `r` is multiple indices, apply to each
    fn not(&mut self, r: Register) -> Register {
        self.real_mat("not", r, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply X to `r`, if `r` is multiple indices, apply to each
    fn x(&mut self, r: Register) -> Register {
        self.real_mat("X", r, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply Y to `r`, if `r` is multiple indices, apply to each
    fn y(&mut self, r: Register) -> Register {
        self.mat(
            "Y",
            r,
            from_tuples(&[(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)]),
        )
        .unwrap()
    }

    /// Apply Z to `r`, if `r` is multiple indices, apply to each
    fn z(&mut self, r: Register) -> Register {
        self.real_mat("Z", r, &[1.0, 0.0, 0.0, -1.0]).unwrap()
    }

    /// Apply H to `r`, if `r` is multiple indices, apply to each
    fn hadamard(&mut self, r: Register) -> Register {
        let inv_sqrt = 1.0f64 / 2.0f64.sqrt();
        self.real_mat("H", r, &[inv_sqrt, inv_sqrt, inv_sqrt, -inv_sqrt])
            .unwrap()
    }

    /// Transforms `|psi>` to `e^{i*theta}|psi>`
    fn phase(&mut self, r: Register, theta: f64) -> Register {
        let phase = Complex { re: 0.0, im: theta }.exp();
        self.mat(
            "Phase",
            r,
            vec![phase, Complex::zero(), Complex::zero(), phase],
        )
        .unwrap()
    }

    /// Apply SWAP to `ra` and `rb`
    fn swap(&mut self, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
        let op = self.make_swap_op(&ra, &rb)?;
        let ra_indices = ra.indices.clone();

        let name = String::from("swap");
        let r = self.merge_with_op(vec![ra, rb], Some((name, op)));
        self.split_absolute(r, ra_indices)
    }

    /// Make an operation from the boxed function `f`. This maps c|`r_in`>|`r_out`> to
    /// c*e^i`theta`|`r_in`>|`r_out` ^ `indx`> where `indx` and `theta` are the outputs from the
    /// function `f(x) = (indx, theta)`
    fn apply_function(
        &mut self,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError>;

    /// Merge the Registers in `rs` into a single Register.
    fn merge(&mut self, rs: Vec<Register>) -> Register {
        self.merge_with_op(rs, None)
    }

    /// Split the Register `r` into two Registers, one with relative `indices` and one with the remaining.
    fn split(
        &mut self,
        r: Register,
        indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError> {
        for indx in &indices {
            if *indx > r.n() {
                let message = format!(
                    "All indices for splitting must be below r.n={:?}, found indx={:?}",
                    r.n(),
                    *indx
                );
                return CircuitError::make_err(message);
            }
        }
        if indices.is_empty() {
            CircuitError::make_str_err("Indices must contain at least one index.")
        } else if indices.len() == r.indices.len() {
            CircuitError::make_str_err("Indices must leave at least one index.")
        } else {
            let selected_indices: Vec<u64> =
                indices.into_iter().map(|i| r.indices[i as usize]).collect();
            self.split_absolute(r, selected_indices)
        }
    }

    /// Split the Register `r` into two Registers, one with `selected_indices` and one with the remaining.
    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError>;

    /// Split the Register into many Registers, each with the given set of indices.
    fn split_absolute_many(
        &mut self,
        r: Register,
        index_groups: Vec<Vec<u64>>,
    ) -> Result<(Vec<Register>, Option<Register>), CircuitError> {
        index_groups
            .into_iter()
            .try_fold((vec![], Some(r)), |(mut rs, r), indices| {
                if let Some(r) = r {
                    if r.indices == indices {
                        rs.push(r);
                        Ok((rs, None))
                    } else {
                        let (hr, tr) = self.split_absolute(r, indices)?;
                        rs.push(hr);
                        Ok((rs, Some(tr)))
                    }
                } else {
                    Ok((rs, None))
                }
            })
    }

    /// Split `r` into a single Register for each index.
    fn split_all(&mut self, r: Register) -> Vec<Register> {
        if r.indices.len() == 1 {
            vec![r]
        } else {
            let mut indices: Vec<Vec<u64>> = r.indices.iter().cloned().map(|i| vec![i]).collect();
            indices.pop();
            // Cannot fail since all indices are from r.
            let (mut rs, r) = self.split_absolute_many(r, indices).unwrap();
            if let Some(r) = r {
                rs.push(r);
            };
            rs
        }
    }

    /// Build a generic matrix op.
    fn make_mat_op(
        &self,
        r: &Register,
        data: Vec<Complex<f64>>,
    ) -> Result<UnitaryOp, CircuitError> {
        make_matrix_op(r.indices.clone(), data)
    }

    /// Build a sparse matrix op
    fn make_sparse_mat_op(
        &self,
        r: &Register,
        data: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<UnitaryOp, CircuitError> {
        make_sparse_matrix_op(r.indices.clone(), data, natural_order)
    }

    /// Build a swap op. ra and rb must have the same number of indices.
    fn make_swap_op(&self, ra: &Register, rb: &Register) -> Result<UnitaryOp, CircuitError> {
        make_swap_op(ra.indices.clone(), rb.indices.clone())
    }

    /// Make a function op. f must be boxed so that this function doesn't need to be parameterized.
    fn make_function_op(
        &self,
        r_in: &Register,
        r_out: &Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<UnitaryOp, CircuitError> {
        make_function_op(r_in.indices.clone(), r_out.indices.clone(), f)
    }

    /// Merge Registers using a generic state processing function.
    fn merge_with_op(
        &mut self,
        rs: Vec<Register>,
        named_operator: Option<(String, UnitaryOp)>,
    ) -> Register;

    /// Measure all Register states and probabilities, does not edit state (thus Unitary). Returns
    /// Register and handle.
    fn stochastic_measure(&mut self, r: Register) -> (Register, u64);

    /// Create a circuit portion which depends on the classical results of measuring some Registers.
    fn single_register_classical_sidechannel(
        &mut self,
        r: Register,
        handles: &[MeasurementHandle],
        f: Box<SingleRegisterSideChannelFn>,
    ) -> Register {
        self.classical_sidechannel(
            vec![r],
            handles,
            Box::new(
                move |b: &mut dyn UnitaryBuilder,
                      mut rs: Vec<Register>,
                      measurements: &[u64]|
                      -> Result<Vec<Register>, CircuitError> {
                    Ok(vec![f(b, rs.pop().unwrap(), measurements)?])
                },
            ),
        )
        .pop()
        .unwrap()
    }

    /// Create a circuit portion which depends on the classical results of measuring some Registers.
    fn classical_sidechannel(
        &mut self,
        rs: Vec<Register>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelFn>,
    ) -> Vec<Register> {
        let index_groups: Vec<_> = rs.iter().map(|r| &r.indices).cloned().collect();
        let f = Box::new(
            move |b: &mut dyn UnitaryBuilder,
                  r: Register,
                  ms: &[u64]|
                  -> Result<Vec<Register>, CircuitError> {
                let (rs, _) = b.split_absolute_many(r, index_groups.clone())?;
                f(b, rs, ms)
            },
        );
        self.sidechannel_helper(rs, handles, f)
    }

    /// A helper function for the classical_sidechannel. Takes a set of Registers to pass to the
    /// subcircuit, a set of handles whose measured values will also be passed, and a function
    /// which matches to description of `SideChannelHelperFn`. Returns a set of Registers whose indices
    /// match those of the input Registers.
    /// This shouldn't be called in circuits, and is a helper to `classical_sidechannel` and
    /// `single_register_classical_sidechannel`.
    fn sidechannel_helper(
        &mut self,
        rs: Vec<Register>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Register>;
}

/// Helper function for Boxing static functions and applying using the given UnitaryBuilder.
pub fn apply_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    r_in: Register,
    r_out: Register,
    f: F,
) -> Result<(Register, Register), CircuitError> {
    b.apply_function(r_in, r_out, Box::new(f))
}

/// Helper function for Boxing static functions and building sparse mats using the given
/// UnitaryBuilder.
pub fn apply_sparse_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    r_in: Register,
    r_out: Register,
    f: F,
) -> Result<(Register, Register), CircuitError> {
    b.apply_function(r_in, r_out, Box::new(f))
}

/// A basic builder for unitary and non-unitary ops.
#[derive(Default, Debug)]
pub struct OpBuilder {
    qubit_index: u64,
    op_id: u64,
    temp_zero_qubits: Vec<Register>,
    temp_one_qubits: Vec<Register>,
}

impl OpBuilder {
    /// Build a new OpBuilder
    pub fn new() -> OpBuilder {
        OpBuilder::default()
    }

    /// Build a new Register with `n` indices
    pub fn register(&mut self, n: u64) -> Result<Register, CircuitError> {
        if n == 0 {
            CircuitError::make_str_err("Register n must be greater than 0.")
        } else {
            let base_index = self.qubit_index;
            self.qubit_index += n;
            Register::new(self.get_op_id(), (base_index..self.qubit_index).collect())
        }
    }

    /// Builds a vector of new Register
    pub fn registers(&mut self, ns: &[u64]) -> Result<Vec<Register>, CircuitError> {
        ns.iter()
            .try_for_each(|n| {
                if *n == 0 {
                    CircuitError::make_str_err("Register n must be greater than 0.")
                } else {
                    Ok(())
                }
            })
            .map(|_| ns.iter().map(|n| self.register(*n).unwrap()).collect())
    }

    /// If you just plan to call unwrap this is cleaner.
    pub fn r(&mut self, n: u64) -> Register {
        self.register(n).unwrap()
    }

    /// Create a single qubit register.
    pub fn qubit(&mut self) -> Register {
        self.r(1)
    }

    /// Build a new register with `n` indices, return it plus a handle which can be
    /// used for feeding in an initial state.
    pub fn register_and_handle(
        &mut self,
        n: u64,
    ) -> Result<(Register, RegisterHandle), CircuitError> {
        let r = self.register(n)?;
        let h = r.handle();
        Ok((r, h))
    }

    /// Get a temporary Register with value `|0n>` or `|1n>`.
    /// This value is not checked and may be subject to the noise of your circuit, since it can
    /// recycle Registers which were returned with `return_temp_register`. If not enough Registers have been
    /// returned, then new Registers may be allocated (and initialized with the correct value).
    pub fn get_temp_register(&mut self, n: u64, value: bool) -> Register {
        let (op_vec, other_vec) = if value {
            (&mut self.temp_one_qubits, &mut self.temp_zero_qubits)
        } else {
            (&mut self.temp_zero_qubits, &mut self.temp_one_qubits)
        };
        let mut acquired_qubits = op_vec.split_off(n as usize);

        // If we didn't get enough, take from temps with the wrong bit.
        if acquired_qubits.len() < n as usize {
            let remaining = n - acquired_qubits.len() as u64;
            let additional_registers = other_vec.split_off(remaining as usize);
            let r = self.merge(additional_registers);
            let r = self.not(r);
            acquired_qubits.push(r);
        }

        // If there still aren't enough, start allocating more (and apply NOT if needed).
        if acquired_qubits.len() < n as usize {
            let remaining = n - acquired_qubits.len() as u64;
            let r = self.register(remaining).unwrap();
            let r = if value { self.not(r) } else { r };
            acquired_qubits.push(r);
        };

        // Make the temp Register.
        self.merge(acquired_qubits)
    }

    /// Return a temporary Register which is supposed to have a given value `|0n>` or `|1n>`
    /// This value is not checked and may be subject to the noise of your circuit, in turn causing
    /// noise to future calls to `get_temp_register`.
    pub fn return_temp_register(&mut self, r: Register, value: bool) {
        let rs = self.split_all(r);
        let op_vec = if value {
            &mut self.temp_one_qubits
        } else {
            &mut self.temp_zero_qubits
        };
        op_vec.extend(rs.into_iter());
    }

    /// Add a measure op to the pipeline for `r` and return a reference which can
    /// later be used to access the measured value from the results of `pipeline::run`.
    pub fn measure(&mut self, r: Register) -> (Register, MeasurementHandle) {
        self.measure_basis(r, 0.0)
    }

    /// Measure in the basis of `cos(phase)|0> + sin(phase)|1>`
    pub fn measure_basis(&mut self, r: Register, angle: f64) -> (Register, MeasurementHandle) {
        let id = self.get_op_id();
        let modifier = StateModifier::new_measurement_basis(
            String::from("measure"),
            id,
            r.indices.clone(),
            angle,
        );
        let modifier = Some(modifier);
        let r = Register::merge_with_modifier(id, vec![r], modifier);
        Register::make_measurement_handle(self.get_op_id(), r)
    }

    fn get_op_id(&mut self) -> u64 {
        let tmp = self.op_id;
        self.op_id += 1;
        tmp
    }
}

impl UnitaryBuilder for OpBuilder {
    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_register: Some(r),
        }
    }

    fn mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<Register, CircuitError> {
        // Special case for broadcasting ops
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| self.mat(name, r, mat.clone()).unwrap())
                .collect();
            Ok(self.merge_with_op(rs, None))
        } else {
            let op = self.make_mat_op(&r, mat)?;
            let name = String::from(name);
            Ok(self.merge_with_op(vec![r], Some((name, op))))
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Register, CircuitError> {
        // Special case for broadcasting ops
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| {
                    self.sparse_mat(name, r, mat.clone(), natural_order)
                        .unwrap()
                })
                .collect();
            Ok(self.merge_with_op(rs, None))
        } else {
            let op = self.make_sparse_mat_op(&r, mat, natural_order)?;
            let name = String::from(name);
            Ok(self.merge_with_op(vec![r], Some((name, op))))
        }
    }

    fn apply_function(
        &mut self,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError> {
        let op = self.make_function_op(&r_in, &r_out, f)?;
        let in_indices = r_in.indices.clone();
        let name = String::from("f");
        let r = self.merge_with_op(vec![r_in, r_out], Some((name, op)));
        self.split_absolute(r, in_indices)
    }

    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError> {
        Register::split_absolute(self.get_op_id(), self.get_op_id(), r, selected_indices)
    }

    fn merge_with_op(
        &mut self,
        rs: Vec<Register>,
        named_operator: Option<(String, UnitaryOp)>,
    ) -> Register {
        let modifier = named_operator.map(|(name, op)| StateModifier::new_unitary(name, op));
        Register::merge_with_modifier(self.get_op_id(), rs, modifier)
    }

    fn stochastic_measure(&mut self, r: Register) -> (Register, u64) {
        let id = self.get_op_id();
        let modifier = StateModifier::new_stochastic_measurement(
            String::from("stochastic"),
            id,
            r.indices.clone(),
        );
        let modifier = Some(modifier);
        let r = Register::merge_with_modifier(id, vec![r], modifier);
        (r, id)
    }

    fn sidechannel_helper(
        &mut self,
        rs: Vec<Register>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Register> {
        let index_groups: Vec<_> = rs.iter().map(|r| &r.indices).cloned().collect();
        let req_qubits = self.qubit_index + 1;

        let f = Box::new(
            move |measurements: &[u64]| -> Result<Vec<StateModifier>, CircuitError> {
                let mut b = Self::new();
                let r = b.register(req_qubits)?;
                let rs = f(&mut b, r, measurements)?;
                let r = b.merge(rs);
                Ok(get_owned_opfns(r))
            },
        );

        let modifier = Some(StateModifier::new_side_channel(
            String::from("SideInputCircuit"),
            handles,
            f,
        ));
        let r = Register::merge_with_modifier(self.get_op_id(), rs, modifier);
        let deps = handles.iter().map(|m| m.clone_register()).collect();
        let r = Register::add_deps(r, deps);
        let (rs, _) = self.split_absolute_many(r, index_groups).unwrap();
        rs
    }
}

/// An op builder which depends on the value of a given Register (COPs)
pub struct ConditionalContextBuilder<'a> {
    parent_builder: &'a mut dyn UnitaryBuilder,
    conditioned_register: Option<Register>,
}

impl<'a> fmt::Debug for ConditionalContextBuilder<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "ConditionalContextBuilder({:?})",
            self.conditioned_register
        )
    }
}

impl<'a> ConditionalContextBuilder<'a> {
    /// Release the Register used to build this builder
    pub fn release_register(self: Self) -> Register {
        match self.conditioned_register {
            Some(r) => r,
            None => panic!("Conditional context builder failed to populate register."),
        }
    }

    fn get_conditional_register(&mut self) -> Register {
        self.conditioned_register.take().unwrap()
    }

    fn set_conditional_register(&mut self, cr: Register) {
        self.conditioned_register = Some(cr);
    }
}

impl<'a> UnitaryBuilder for ConditionalContextBuilder<'a> {
    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_register: Some(r),
        }
    }

    fn mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<Register, CircuitError> {
        // Special case for applying mat to each Register in collection.
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| self.mat(name, r, mat.clone()).unwrap())
                .collect();
            Ok(self.merge_with_op(rs, None))
        } else {
            let op = self.make_mat_op(&r, mat)?;
            let cr = self.get_conditional_register();
            let cr_indices = cr.indices.clone();
            let name = format!("C({})", name);
            let r = self.merge_with_op(vec![cr, r], Some((name, op)));
            let (cr, r) = self.split_absolute(r, cr_indices).unwrap();

            self.set_conditional_register(cr);
            Ok(r)
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<Register, CircuitError> {
        // Special case for applying mat to each Register in collection.
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| {
                    self.sparse_mat(name, r, mat.clone(), natural_order)
                        .unwrap()
                })
                .collect();
            Ok(self.merge_with_op(rs, None))
        } else {
            let op = self.make_sparse_mat_op(&r, mat, natural_order)?;
            let cr = self.get_conditional_register();
            let cr_indices = cr.indices.clone();
            let name = format!("C({})", name);
            let r = self.merge_with_op(vec![cr, r], Some((name, op)));
            let (cr, r) = self.split_absolute(r, cr_indices).unwrap();

            self.set_conditional_register(cr);
            Ok(r)
        }
    }

    fn swap(&mut self, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
        let op = self.make_swap_op(&ra, &rb)?;
        let cr = self.get_conditional_register();
        let cr_indices = cr.indices.clone();
        let ra_indices = ra.indices.clone();
        let name = String::from("C(swap)");
        let r = self.merge_with_op(vec![cr, ra, rb], Some((name, op)));
        let (cr, r) = self.split_absolute(r, cr_indices).unwrap();
        let (ra, rb) = self.split_absolute(r, ra_indices).unwrap();

        self.set_conditional_register(cr);
        Ok((ra, rb))
    }

    fn apply_function(
        &mut self,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError> {
        let op = self.make_function_op(&r_in, &r_out, f)?;
        let cr = self.get_conditional_register();

        let cr_indices = cr.indices.clone();
        let in_indices = r_in.indices.clone();
        let name = String::from("C(f)");
        let r = self.merge_with_op(vec![cr, r_in, r_out], Some((name, op)));
        let (cr, r) = self.split_absolute(r, cr_indices).unwrap();
        let (r_in, r_out) = self.split_absolute(r, in_indices).unwrap();

        self.set_conditional_register(cr);
        Ok((r_in, r_out))
    }

    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: Vec<u64>,
    ) -> Result<(Register, Register), CircuitError> {
        self.parent_builder.split_absolute(r, selected_indices)
    }

    fn make_mat_op(
        &self,
        r: &Register,
        data: Vec<Complex<f64>>,
    ) -> Result<UnitaryOp, CircuitError> {
        match &self.conditioned_register {
            Some(cr) => make_control_op(
                cr.indices.clone(),
                self.parent_builder.make_mat_op(r, data)?,
            ),
            None => panic!("Conditional context builder failed to populate Register."),
        }
    }

    fn make_sparse_mat_op(
        &self,
        r: &Register,
        data: Vec<Vec<(u64, Complex<f64>)>>,
        natural_order: bool,
    ) -> Result<UnitaryOp, CircuitError> {
        match &self.conditioned_register {
            Some(cr) => make_control_op(
                cr.indices.clone(),
                self.parent_builder
                    .make_sparse_mat_op(r, data, natural_order)?,
            ),
            None => panic!("Conditional context builder failed to populate Register."),
        }
    }

    fn make_swap_op(&self, ra: &Register, rb: &Register) -> Result<UnitaryOp, CircuitError> {
        match &self.conditioned_register {
            Some(cr) => {
                let op = self.parent_builder.make_swap_op(ra, rb)?;
                make_control_op(cr.indices.clone(), op)
            }
            None => panic!("Conditional context builder failed to populate Register."),
        }
    }

    fn make_function_op(
        &self,
        r_in: &Register,
        r_out: &Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<UnitaryOp, CircuitError> {
        match &self.conditioned_register {
            Some(cr) => {
                let op = self.parent_builder.make_function_op(r_in, r_out, f)?;
                make_control_op(cr.indices.clone(), op)
            }
            None => panic!("Conditional context builder failed to populate Register."),
        }
    }

    fn merge_with_op(
        &mut self,
        rs: Vec<Register>,
        named_operator: Option<(String, UnitaryOp)>,
    ) -> Register {
        self.parent_builder.merge_with_op(rs, named_operator)
    }

    fn stochastic_measure(&mut self, r: Register) -> (Register, u64) {
        self.parent_builder.stochastic_measure(r)
    }

    fn sidechannel_helper(
        &mut self,
        rs: Vec<Register>,
        handles: &[MeasurementHandle],
        f: Box<SideChannelHelperFn>,
    ) -> Vec<Register> {
        let cr = self.get_conditional_register();
        let conditioned_indices = cr.indices.clone();
        let cindices_clone = conditioned_indices.clone();
        let f = Box::new(
            move |b: &mut dyn UnitaryBuilder,
                  r: Register,
                  ms: &[u64]|
                  -> Result<Vec<Register>, CircuitError> {
                let (cr, r) = b.split_absolute(r, conditioned_indices.clone())?;
                let mut b = b.with_condition(cr);
                f(&mut b, r, ms)
            },
        );
        let rs = self.parent_builder.sidechannel_helper(rs, handles, f);
        let index_groups: Vec<_> = rs.iter().map(|r| r.indices.clone()).collect();
        let r = self.merge(rs);
        let r = self.merge(vec![cr, r]);
        let (cr, r) = self.split_absolute(r, cindices_clone).unwrap();
        self.set_conditional_register(cr);
        let (rs, _) = self.split_absolute_many(r, index_groups).unwrap();
        rs
    }
}
