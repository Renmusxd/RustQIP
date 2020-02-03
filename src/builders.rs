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
    /// Create a single qubit register.
    fn qubit(&mut self) -> Register;

    /// Build a new Register with `n` indices
    fn register(&mut self, n: u64) -> Result<Register, CircuitError> {
        if n == 0 {
            CircuitError::make_str_err("Register n must be greater than 0.")
        } else {
            let rs = (0..n).map(|_| self.qubit()).collect();
            self.merge(rs)
        }
    }

    /// Builds a vector of new Register
    fn registers(&mut self, ns: &[u64]) -> Result<Vec<Register>, CircuitError> {
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

    /// Build a new register with `n` indices, return it plus a handle which can be
    /// used for feeding in an initial state.
    fn register_and_handle(&mut self, n: u64) -> Result<(Register, RegisterHandle), CircuitError> {
        let r = self.register(n)?;
        let h = r.handle();
        Ok((r, h))
    }

    /// Get a temporary Register with value `|0n>` or `|1n>`.
    /// This value is not checked and may be subject to the noise of your circuit, since it can
    /// recycle Registers which were returned with `return_temp_register`. If not enough Registers have been
    /// returned, then new Registers may be allocated (and initialized with the correct value).
    fn get_temp_register(&mut self, n: u64, value: bool) -> Register;

    /// Return a temporary Register which is supposed to have a given value `|0n>` or `|1n>`
    /// This value is not checked and may be subject to the noise of your circuit, in turn causing
    /// noise to future calls to `get_temp_register`.
    fn return_temp_register(&mut self, r: Register, value: bool);

    /// Build a builder which uses `r` as a condition.
    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder;

    /// Add a name scope.
    fn push_name_scope(&mut self, name: &str);

    /// Remove and return a name scope.
    fn pop_name_scope(&mut self) -> Option<String>;

    /// Get list of names in above scopes.
    fn get_name_list(&self) -> &[String];

    /// Get the full name with scope
    fn get_full_name(&self, name: &str) -> String {
        let names = self.get_name_list();
        if names.is_empty() {
            name.to_string()
        } else {
            format!("{}/{}", names.join("/"), name)
        }
    }

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
        order: Representation,
    ) -> Result<Register, CircuitError>;

    /// Build a sparse matrix op from `f`, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^n).
    fn sparse_mat_from_fn(
        &mut self,
        name: &str,
        r: Register,
        f: Box<SparseBuilderFn>,
        order: Representation,
    ) -> Result<Register, CircuitError> {
        let n = r.indices.len();
        let mat = make_sparse_matrix_from_function(n, f, order);
        self.sparse_mat(name, r, mat, Representation::BigEndian)
    }

    /// Build a sparse matrix op from real numbers, apply to `r`, if `r` is multiple indices and
    /// mat is 2x2, apply to each index, otherwise returns an error if the matrix is not the correct
    /// size for the number of indices in `r` (mat.len() == 2^n).
    fn real_sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: &[Vec<(u64, f64)>],
        order: Representation,
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
        self.sparse_mat(name, r, mat, order)
    }

    /// Apply NOT to `r`, if `r` is multiple indices, apply to each
    fn not(&mut self, r: Register) -> Register {
        self.real_mat("not", r, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply X to `r`, if `r` is multiple indices, apply to each
    fn x(&mut self, r: Register) -> Register {
        self.real_mat("X", r, &[0.0, 1.0, 1.0, 0.0]).unwrap()
    }

    /// Apply Rx to `r`, if `r` is multiple indices, apply to each
    /// * `theta` - the angle to rotate around the x axis of the Bloch sphere
    fn rx(&mut self, r: Register, theta: f64) -> Register {
        let theta_2 = theta / 2.0;
        let (sin, cos) = theta_2.sin_cos();
        self.mat(
            "Rx",
            r,
            from_tuples(&[(cos, 0.0), (0.0, -sin), (0.0, -sin), (cos, 0.0)]),
        )
        .unwrap()
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

    /// Apply Ry to `r`, if `r` is multiple indices, apply to each
    /// * `theta` - the angle to rotate around the y axis of the Bloch sphere
    fn ry(&mut self, r: Register, theta: f64) -> Register {
        let theta_2 = theta / 2.0;
        let (sin, cos) = theta_2.sin_cos();
        self.real_mat("Ry", r, &[cos, -sin, sin, cos]).unwrap()
    }

    /// Apply Z to `r`, if `r` is multiple indices, apply to each
    fn z(&mut self, r: Register) -> Register {
        self.real_mat("Z", r, &[1.0, 0.0, 0.0, -1.0]).unwrap()
    }

    /// Apply Rz to `r`, if `r` is multiple indices, apply to each
    /// * `theta` - the angle to rotate around the z axis of the Bloch sphere
    fn rz(&mut self, r: Register, theta: f64) -> Register {
        let theta_2 = theta / 2.0;
        let phase_plus = Complex {
            re: 0.0,
            im: theta_2,
        }
        .exp();
        let phase_minus = Complex {
            re: 0.0,
            im: -theta_2,
        }
        .exp();
        self.mat(
            "Rz",
            r,
            vec![phase_minus, Complex::zero(), Complex::zero(), phase_plus],
        )
        .unwrap()
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
        let r = self.merge_with_op(vec![ra, rb], Some(("swap".to_string(), op)))?;
        let (ra, rb) = self.split_absolute(r, &ra_indices)?;
        Ok((ra, rb.unwrap()))
    }

    /// Make an operation from the boxed function `f`. This maps c|`r_in`>|`r_out`> to
    /// c*e^i`theta`|`r_in`>|`r_out` ^ `indx`> where `indx` and `theta` are the outputs from the
    /// function `f(x) = (indx, theta)`
    fn apply_function(
        &mut self,
        name: &str,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError>;

    /// A controlled x, using `cr` as control and `r` as input.
    fn cx(&mut self, cr: Register, r: Register) -> (Register, Register) {
        let mut b = self.with_condition(cr);
        let r = b.x(r);
        let cr = b.release_register();
        (cr, r)
    }
    /// A controlled y, using `cr` as control and `r` as input.
    fn cy(&mut self, cr: Register, r: Register) -> (Register, Register) {
        let mut b = self.with_condition(cr);
        let r = b.y(r);
        let cr = b.release_register();
        (cr, r)
    }
    /// A controlled z, using `cr` as control and `r` as input.
    fn cz(&mut self, cr: Register, r: Register) -> (Register, Register) {
        let mut b = self.with_condition(cr);
        let r = b.z(r);
        let cr = b.release_register();
        (cr, r)
    }
    /// A controlled not, using `cr` as control and `r` as input.
    fn cnot(&mut self, cr: Register, r: Register) -> (Register, Register) {
        let mut b = self.with_condition(cr);
        let r = b.not(r);
        let cr = b.release_register();
        (cr, r)
    }
    /// Swap `ra` and `rb` controlled by `cr`.
    fn cswap(
        &mut self,
        cr: Register,
        ra: Register,
        rb: Register,
    ) -> Result<(Register, Register, Register), CircuitError> {
        let mut b = self.with_condition(cr);
        let (ra, rb) = b.swap(ra, rb)?;
        let cr = b.release_register();
        Ok((cr, ra, rb))
    }
    /// Apply a unitary matrix to the register. If mat is 2x2 then can broadcast to all qubits.
    fn cmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: Vec<Complex<f64>>,
    ) -> Result<(Register, Register), CircuitError> {
        let mut b = self.with_condition(cr);
        let r = b.mat(name, r, mat)?;
        let cr = b.release_register();
        Ok((cr, r))
    }
    /// Apply a orthonormal matrix to the register. If mat is 2x2 then can broadcast to all qubits.
    fn crealmat(
        &mut self,
        name: &str,
        cr: Register,
        r: Register,
        mat: &[f64],
    ) -> Result<(Register, Register), CircuitError> {
        let mut b = self.with_condition(cr);
        let r = b.real_mat(name, r, mat)?;
        let cr = b.release_register();
        Ok((cr, r))
    }

    /// Merge the Registers in `rs` into a single Register.
    fn merge(&mut self, rs: Vec<Register>) -> Result<Register, CircuitError> {
        self.merge_with_op(rs, None)
    }

    /// Split the Register `r` into two Registers, one with relative `indices` and one with the remaining.
    fn split(
        &mut self,
        r: Register,
        indices: &[u64],
    ) -> Result<(Register, Option<Register>), CircuitError> {
        for indx in indices {
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
        } else {
            let selected_indices: Vec<u64> =
                indices.iter().map(|i| r.indices[(*i) as usize]).collect();
            self.split_absolute(r, &selected_indices)
        }
    }

    /// Split the Register `r` into two Registers, one with `selected_indices` and one with the remaining.
    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: &[u64],
    ) -> Result<(Register, Option<Register>), CircuitError>;

    /// Split the Register into many Registers, each with the given set of indices.
    fn split_absolute_many(
        &mut self,
        r: Register,
        index_groups: &[Vec<u64>],
    ) -> Result<(Vec<Register>, Option<Register>), CircuitError> {
        index_groups
            .iter()
            .try_fold((vec![], Some(r)), |(mut rs, r), indices| {
                if let Some(r) = r {
                    let (hr, tr) = self.split_absolute(r, indices)?;
                    rs.push(hr);
                    Ok((rs, tr))
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
            let (mut rs, r) = self.split_absolute_many(r, &indices).unwrap();
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
        order: Representation,
    ) -> Result<UnitaryOp, CircuitError> {
        make_sparse_matrix_op(r.indices.clone(), data, order)
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
    ) -> Result<Register, CircuitError>;

    /// Merge a set of qubits into a given qubit at a set of indices
    fn merge_with_indices(
        &mut self,
        into: Register,
        qubits: Vec<Register>,
        at_indices: &[u64],
    ) -> Result<Register, CircuitError> {
        if qubits.len() != at_indices.len() {
            CircuitError::make_err(format!(
                "Number of qubits ({}) must equal number of indices ({}).",
                qubits.len(),
                at_indices.len()
            ))
        } else {
            let zipped = qubits.into_iter().zip(at_indices.iter().cloned()).collect();
            self.merge_with(into, zipped)
        }
    }

    /// Merge a set of qubits into a given qubit at a set of indices
    fn merge_with(
        &mut self,
        into: Register,
        qubit_and_index: Vec<(Register, u64)>,
    ) -> Result<Register, CircuitError> {
        let total_n = into.n() + qubit_and_index.len() as u64;
        qubit_and_index.iter().enumerate().try_for_each(|(indx, (r,r_indx))| {
            if r.n() == 1 {
                if *r_indx >= total_n {
                    CircuitError::make_err(format!("Attempting to insert a register (#[{}]) at index={} when total size is {}: ", indx, r_indx, total_n))
                } else {
                    Ok(())
                }
            } else {
                CircuitError::make_err(format!("Attempting to insert a register (#[{}]) with multiple qubits (#{}) at a single index: ", indx, r.n()))
            }
        })?;
        let mut into_qs = self.split_all(into);
        let mut insert_qs = qubit_and_index;
        into_qs.reverse();
        insert_qs.reverse();
        let init_v = Vec::with_capacity(total_n as usize);
        let qs = (0..total_n).fold(init_v, |mut acc, indx| {
            if let Some((last_r, last_indx)) = insert_qs.pop() {
                if last_indx == indx {
                    acc.push(last_r);
                } else {
                    insert_qs.push((last_r, last_indx));
                    if let Some(into_r) = into_qs.pop() {
                        acc.push(into_r)
                    } else {
                        panic!("This shouldn't happen");
                    }
                }
            } else if let Some(into_r) = into_qs.pop() {
                acc.push(into_r)
            } else {
                panic!("This shouldn't happen");
            }
            acc
        });
        self.merge(qs)
    }

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
                let (rs, _) = b.split_absolute_many(r, &index_groups)?;
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

    /// Debug a register using a function `f` run during circuit execution on the state of `r`
    fn debug(
        &mut self,
        r: Register,
        f: Box<dyn Fn(Vec<f64>) -> ()>,
    ) -> Result<Register, CircuitError> {
        let vf = Box::new(move |mut states: Vec<Vec<f64>>| {
            let state = states.pop().unwrap();
            f(state);
        });
        let mut rs = self.debug_registers(vec![r], vf)?;
        Ok(rs.pop().unwrap())
    }

    /// Debug a vec of registers using a function `f` run during circuit execution on each state of `r`
    fn debug_registers(
        &mut self,
        rs: Vec<Register>,
        f: Box<dyn Fn(Vec<Vec<f64>>) -> ()>,
    ) -> Result<Vec<Register>, CircuitError>;
}

/// Helper function for Boxing static functions and applying using the given UnitaryBuilder.
pub fn apply_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    r_in: Register,
    r_out: Register,
    f: F,
) -> Result<(Register, Register), CircuitError> {
    b.apply_function("f", r_in, r_out, Box::new(f))
}

/// Helper function for Boxing static functions and building sparse mats using the given
/// UnitaryBuilder.
pub fn apply_sparse_function<F: 'static + Fn(u64) -> (u64, f64) + Send + Sync>(
    b: &mut dyn UnitaryBuilder,
    r_in: Register,
    r_out: Register,
    f: F,
) -> Result<(Register, Register), CircuitError> {
    b.apply_function("f", r_in, r_out, Box::new(f))
}

/// A basic builder for unitary and non-unitary ops.
#[derive(Default, Debug)]
pub struct OpBuilder {
    qubit_index: u64,
    op_id: u64,
    temp_zero_qubits: Vec<Register>,
    temp_one_qubits: Vec<Register>,
    names: Vec<String>,
}

impl OpBuilder {
    /// Build a new OpBuilder
    pub fn new() -> OpBuilder {
        OpBuilder::default()
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
        let r = Register::merge_with_modifier(id, vec![r], modifier).unwrap();
        Register::make_measurement_handle(self.get_op_id(), r)
    }

    /// Get the current count of created qubits.
    pub fn get_qubit_count(&self) -> u64 {
        self.qubit_index
    }

    /// Get indices of zeros and ones temps currently in holding.
    pub(crate) fn get_temp_indices(&self) -> (Vec<u64>, Vec<u64>) {
        let zeros = self
            .temp_zero_qubits
            .iter()
            .map(|r| &r.indices)
            .flatten()
            .cloned()
            .collect();
        let ones = self
            .temp_one_qubits
            .iter()
            .map(|r| &r.indices)
            .flatten()
            .cloned()
            .collect();
        (zeros, ones)
    }

    fn get_op_id(&mut self) -> u64 {
        let tmp = self.op_id;
        self.op_id += 1;
        tmp
    }
}

impl UnitaryBuilder for OpBuilder {
    fn qubit(&mut self) -> Register {
        let base_index = self.qubit_index;
        self.qubit_index += 1;
        Register::new(self.get_op_id(), vec![base_index]).unwrap()
    }

    fn register(&mut self, n: u64) -> Result<Register, CircuitError> {
        if n == 0 {
            CircuitError::make_str_err("Register n must be greater than 0.")
        } else {
            let base_index = self.qubit_index;
            self.qubit_index += n;
            Register::new(self.get_op_id(), (base_index..self.qubit_index).collect())
        }
    }

    fn get_temp_register(&mut self, n: u64, value: bool) -> Register {
        let (op_vec, other_vec) = if value {
            (&mut self.temp_one_qubits, &mut self.temp_zero_qubits)
        } else {
            (&mut self.temp_zero_qubits, &mut self.temp_one_qubits)
        };

        let n = n as usize;
        let mut acquired_qubits = if n < op_vec.len() {
            op_vec.split_off(op_vec.len() - n)
        } else if !op_vec.is_empty() {
            op_vec.split_off(op_vec.len() - 1)
        } else {
            vec![]
        };

        // If we didn't get enough, take from temps with the wrong bit.
        if acquired_qubits.len() < n {
            let remaining = n - acquired_qubits.len();
            let additional_registers = if remaining < other_vec.len() {
                other_vec.split_off(other_vec.len() - remaining)
            } else if !other_vec.is_empty() {
                other_vec.split_off(other_vec.len() - 1)
            } else {
                vec![]
            };
            if !additional_registers.is_empty() {
                let r = self.merge(additional_registers).unwrap();
                let r = self.not(r);
                let rs = self.split_all(r);
                acquired_qubits.extend(rs.into_iter());
            }
        }

        // If there still aren't enough, start allocating more (and apply NOT if needed).
        if acquired_qubits.len() < n {
            let remaining = (n - acquired_qubits.len()) as u64;
            let r = self.register(remaining).unwrap();
            let r = if value { self.not(r) } else { r };
            acquired_qubits.push(r);
        };

        // Make the temp Register.
        self.merge(acquired_qubits).unwrap()
    }

    fn return_temp_register(&mut self, r: Register, value: bool) {
        let rs = self.split_all(r);
        let op_vec = if value {
            &mut self.temp_one_qubits
        } else {
            &mut self.temp_zero_qubits
        };
        op_vec.extend(rs.into_iter());
    }

    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_register: Some(r),
        }
    }

    fn push_name_scope(&mut self, name: &str) {
        self.names.push(name.to_string())
    }

    fn pop_name_scope(&mut self) -> Option<String> {
        self.names.pop()
    }

    fn get_name_list(&self) -> &[String] {
        self.names.as_slice()
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
            self.merge_with_op(rs, None)
        } else {
            let op = self.make_mat_op(&r, mat)?;
            self.merge_with_op(vec![r], Some((name.to_string(), op)))
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        order: Representation,
    ) -> Result<Register, CircuitError> {
        // Special case for broadcasting ops
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| self.sparse_mat(name, r, mat.clone(), order).unwrap())
                .collect();
            self.merge_with_op(rs, None)
        } else {
            let op = self.make_sparse_mat_op(&r, mat, order)?;
            self.merge_with_op(vec![r], Some((name.to_string(), op)))
        }
    }

    fn apply_function(
        &mut self,
        name: &str,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError> {
        let op = self.make_function_op(&r_in, &r_out, f)?;
        let in_indices = r_in.indices.clone();
        let r = self.merge_with_op(vec![r_in, r_out], Some((name.to_string(), op)))?;
        let (r_in, r_out) = self.split_absolute(r, &in_indices)?;
        Ok((r_in, r_out.unwrap()))
    }

    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: &[u64],
    ) -> Result<(Register, Option<Register>), CircuitError> {
        Register::split_absolute(self.get_op_id(), self.get_op_id(), r, selected_indices)
    }

    fn merge_with_op(
        &mut self,
        rs: Vec<Register>,
        named_operator: Option<(String, UnitaryOp)>,
    ) -> Result<Register, CircuitError> {
        let modifier = named_operator
            .map(|(name, op)| (self.get_full_name(&name), op))
            .map(|(name, op)| StateModifier::new_unitary(name, op));
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
        let r = Register::merge_with_modifier(id, vec![r], modifier).unwrap();
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
                let r = b.merge(rs)?;
                Ok(get_owned_opfns(r))
            },
        );

        let modifier = Some(StateModifier::new_side_channel(
            String::from("SideInputCircuit"),
            handles,
            f,
        ));
        let r = Register::merge_with_modifier(self.get_op_id(), rs, modifier).unwrap();
        let deps = handles.iter().map(|m| m.clone_register()).collect();
        let r = Register::add_deps(r, deps);
        let (rs, _) = self.split_absolute_many(r, &index_groups).unwrap();
        rs
    }

    fn debug_registers(
        &mut self,
        rs: Vec<Register>,
        f: Box<dyn Fn(Vec<Vec<f64>>) -> ()>,
    ) -> Result<Vec<Register>, CircuitError> {
        let indices: Vec<Vec<u64>> = rs.iter().map(|r| r.indices.clone()).collect();
        let modifier = StateModifier::new_debug("Debug".to_string(), indices.clone(), f);
        let r = Register::merge_with_modifier(self.get_op_id(), rs, Some(modifier))?;
        let (rs, remaining) = self.split_absolute_many(r, &indices)?;
        assert_eq!(remaining, None);
        Ok(rs)
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
    fn qubit(&mut self) -> Register {
        self.parent_builder.qubit()
    }

    fn register(&mut self, n: u64) -> Result<Register, CircuitError> {
        self.parent_builder.register(n)
    }

    fn get_temp_register(&mut self, n: u64, value: bool) -> Register {
        self.parent_builder.get_temp_register(n, value)
    }

    fn return_temp_register(&mut self, r: Register, value: bool) {
        self.parent_builder.return_temp_register(r, value)
    }

    fn with_condition(&mut self, r: Register) -> ConditionalContextBuilder {
        ConditionalContextBuilder {
            parent_builder: self,
            conditioned_register: Some(r),
        }
    }

    fn push_name_scope(&mut self, name: &str) {
        self.parent_builder.push_name_scope(name)
    }

    fn pop_name_scope(&mut self) -> Option<String> {
        self.parent_builder.pop_name_scope()
    }

    fn get_name_list(&self) -> &[String] {
        self.parent_builder.get_name_list()
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
            self.merge_with_op(rs, None)
        } else {
            let op = self.make_mat_op(&r, mat)?;
            let r = self.merge_with_op(vec![r], Some((name.to_string(), op)))?;
            Ok(r)
        }
    }

    fn sparse_mat(
        &mut self,
        name: &str,
        r: Register,
        mat: Vec<Vec<(u64, Complex<f64>)>>,
        order: Representation,
    ) -> Result<Register, CircuitError> {
        // Special case for applying mat to each Register in collection.
        if r.indices.len() > 1 && mat.len() == (2 * 2) {
            let rs = self.split_all(r);
            let rs = rs
                .into_iter()
                .map(|r| self.sparse_mat(name, r, mat.clone(), order).unwrap())
                .collect();
            self.merge_with_op(rs, None)
        } else {
            let op = self.make_sparse_mat_op(&r, mat, order)?;
            let r = self.merge_with_op(vec![r], Some((name.to_string(), op)))?;
            Ok(r)
        }
    }

    fn swap(&mut self, ra: Register, rb: Register) -> Result<(Register, Register), CircuitError> {
        let op = self.make_swap_op(&ra, &rb)?;
        let ra_indices = ra.indices.clone();
        let r = self.merge_with_op(vec![ra, rb], Some(("swap".to_string(), op)))?;
        let (ra, rb) = self.split_absolute(r, &ra_indices)?;
        Ok((ra, rb.unwrap()))
    }

    fn apply_function(
        &mut self,
        name: &str,
        r_in: Register,
        r_out: Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<(Register, Register), CircuitError> {
        let op = self.make_function_op(&r_in, &r_out, f)?;
        let in_indices = r_in.indices.clone();
        let r = self.merge_with_op(vec![r_in, r_out], Some((name.to_string(), op)))?;
        let (r_in, r_out) = self.split_absolute(r, &in_indices)?;
        Ok((r_in, r_out.unwrap()))
    }

    fn split_absolute(
        &mut self,
        r: Register,
        selected_indices: &[u64],
    ) -> Result<(Register, Option<Register>), CircuitError> {
        self.parent_builder.split_absolute(r, selected_indices)
    }

    fn make_mat_op(
        &self,
        r: &Register,
        data: Vec<Complex<f64>>,
    ) -> Result<UnitaryOp, CircuitError> {
        self.parent_builder.make_mat_op(r, data)
    }

    fn make_sparse_mat_op(
        &self,
        r: &Register,
        data: Vec<Vec<(u64, Complex<f64>)>>,
        order: Representation,
    ) -> Result<UnitaryOp, CircuitError> {
        self.parent_builder.make_sparse_mat_op(r, data, order)
    }

    fn make_swap_op(&self, ra: &Register, rb: &Register) -> Result<UnitaryOp, CircuitError> {
        self.parent_builder.make_swap_op(ra, rb)
    }

    fn make_function_op(
        &self,
        r_in: &Register,
        r_out: &Register,
        f: Box<dyn Fn(u64) -> (u64, f64) + Send + Sync>,
    ) -> Result<UnitaryOp, CircuitError> {
        self.parent_builder.make_function_op(r_in, r_out, f)
    }

    fn merge_with_op(
        &mut self,
        mut rs: Vec<Register>,
        named_operator: Option<(String, UnitaryOp)>,
    ) -> Result<Register, CircuitError> {
        if let Some((name, op)) = named_operator {
            let cr = self.get_conditional_register();
            let cr_indices = cr.indices.clone();
            let name = if let UnitaryOp::Control(_, _, _) = &op {
                name
            } else {
                format!("C({})", name)
            };
            let op = make_control_op(cr_indices.clone(), op)?;
            rs.insert(0, cr);
            let r = self.parent_builder.merge_with_op(rs, Some((name, op)))?;
            let (cr, r) = self.split_absolute(r, &cr_indices)?;
            self.set_conditional_register(cr);
            Ok(r.unwrap())
        } else {
            self.parent_builder.merge_with_op(rs, None)
        }
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
                let (cr, r) = b.split_absolute(r, &conditioned_indices)?;
                let r = r.unwrap();
                let mut b = b.with_condition(cr);
                f(&mut b, r, ms)
            },
        );
        let rs = self.parent_builder.sidechannel_helper(rs, handles, f);
        let index_groups: Vec<_> = rs.iter().map(|r| r.indices.clone()).collect();
        let r = self.merge(rs).unwrap();
        let r = self.merge(vec![cr, r]).unwrap();
        let (cr, r) = self.split_absolute(r, &cindices_clone).unwrap();
        self.set_conditional_register(cr);
        let (rs, remaining) = self.split_absolute_many(r.unwrap(), &index_groups).unwrap();
        if remaining.is_some() {
            panic!("There should be no qubits remaining.");
        }
        rs
    }

    fn debug_registers(
        &mut self,
        rs: Vec<Register>,
        f: Box<dyn Fn(Vec<Vec<f64>>) -> ()>,
    ) -> Result<Vec<Register>, CircuitError> {
        self.parent_builder.debug_registers(rs, f)
    }
}

/// Condition a circuit defined by `f` using `cr`.
///
/// # Example
/// ```
/// use qip::*;
///
/// let mut b = qip::OpBuilder::new();
/// let qa = b.qubit();
/// let qb = b.qubit();
/// let qc = b.qubit();
///
/// // Run this subcircuit entirely conditioned on qa, with argument qb.
/// let (qa, qb) = condition(&mut b, qa, qb, |b, q| {
///     // Here qb is bound to q, qa is implicitly a condition on the operation.
///     b.not(q)
/// });
///
/// // We can provide any type as the argument to the function, so long as we can return the same
/// // type. Here we provide the tuple of (Register, Register). This circuit is the same as:
/// // let (qa, qb) = b.cnot(qa, qb);
/// // let (qa, qc) = b.cnot(qa, qc);
/// let (qa, (qb, qc)) = condition(&mut b, qa, (qb, qc), |b, (q1, q2)| {
///     let q1 = b.not(q1);
///     let q2 = b.not(q2);
///     (q1, q2)
/// });
/// ```
pub fn condition<F, RS, OS>(
    b: &mut dyn UnitaryBuilder,
    cr: Register,
    rs: RS,
    f: F,
) -> (Register, OS)
where
    F: FnOnce(&mut dyn UnitaryBuilder, RS) -> OS,
{
    let mut c = b.with_condition(cr);
    let rs = f(&mut c, rs);
    let r = c.release_register();
    (r, rs)
}

/// Condition a circuit defined by `f` using `cr`, better supports Result types.
///
/// # Example
/// ```
/// use qip::*;
/// # fn main() -> Result<(), CircuitError> {
/// let mut b = qip::OpBuilder::new();
/// let qa = b.qubit();
/// let qb = b.qubit();
/// let qc = b.qubit();
///
/// // We can provide any type as the argument to the function, so long as we can return the same
/// // type. Here we provide the tuple of (Register, Register). This circuit is the same as
/// // b.cswap(qa, qb, qc)?;
/// let (qa, (qb, qc)) = try_condition(&mut b, qa, (qb, qc), |b, (q1, q2)| {
///     b.swap(q1, q2)
/// })?;
/// # Ok(())
/// # }
/// ```
pub fn try_condition<F, RS, OS>(
    b: &mut dyn UnitaryBuilder,
    cr: Register,
    rs: RS,
    f: F,
) -> Result<(Register, OS), CircuitError>
where
    F: FnOnce(&mut dyn UnitaryBuilder, RS) -> Result<OS, CircuitError>,
{
    let mut c = b.with_condition(cr);
    let rs = f(&mut c, rs)?;
    let r = c.release_register();
    Ok((r, rs))
}
