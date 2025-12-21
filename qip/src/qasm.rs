//! OpenQASM 2.0 export utilities.

use std::collections::HashMap;
use std::fmt::Write;
use std::fs::File;
use std::io::Write as IoWrite;
use std::path::Path;

use num_rational::Ratio;
use num_traits::ToPrimitive;

use crate::builder::{
    BuilderCircuitObjectType, LocalBuilder, MeasurementObject, RotationObject, UnitaryMatrixObject,
};
use crate::builder_traits::{CircuitBuilder, Subcircuitable};
use crate::types::Precision;

/// Exports a circuit to OpenQASM 2.0 text.
pub trait ToOpenQasm {
    /// Returns the circuit as OpenQASM 2.0 text.
    fn to_openqasm(&self) -> String;
}

impl<P> ToOpenQasm for LocalBuilder<P>
where
    P: Precision + ToPrimitive,
{
    fn to_openqasm(&self) -> String {
        let n_qubits = self.n();
        let pipeline = self
            .make_subcircuit()
            .expect("LocalBuilder::make_subcircuit should not fail");

        // Classical register size: one bit per collapsed measurement.
        let mut measured: Vec<usize> = pipeline
            .iter()
            .filter_map(|(indices, obj)| match obj.object() {
                BuilderCircuitObjectType::Measurement(MeasurementObject::Measurement) => {
                    Some(indices.as_slice())
                }
                _ => None,
            })
            .flat_map(|indices| indices.iter().copied())
            .collect();
        measured.sort_unstable();
        measured.dedup();
        let creg_size = measured.len();
        let classical_map: HashMap<usize, usize> = measured
            .iter()
            .enumerate()
            .map(|(c, q)| (*q, c))
            .collect();

        let mut out = String::new();
        // Header
        writeln!(&mut out, "OPENQASM 2.0;").unwrap();
        writeln!(&mut out, "include \"qelib1.inc\";").unwrap();

        // Registers
        writeln!(&mut out, "qreg q[{}];", n_qubits).unwrap();
        if creg_size > 0 {
            writeln!(&mut out, "creg c[{}];", creg_size).unwrap();
        }

        // Emit ops in pipeline order.
        for (indices, obj) in &pipeline {
            match obj.object() {
                BuilderCircuitObjectType::Unitary(u) => {
                    emit_unitary(u, indices, &mut out);
                }
                BuilderCircuitObjectType::Measurement(m) => match m {
                    // Map each measured qubit -> one classical bit
                    MeasurementObject::Measurement => {
                        for &q in indices {
                            if let Some(&c_index) = classical_map.get(&q) {
                                writeln!(&mut out, "measure q[{}] -> c[{}];", q, c_index).unwrap();
                            }
                        }
                    }
                    // Not representable in OpenQASM 2.0; comment only.
                    MeasurementObject::StochasticMeasurement => {
                        writeln!(
                            &mut out,
                            "// stochastic measurement over {:?} (not in OpenQASM 2.0)",
                            indices
                        )
                        .unwrap();
                    }
                },
            }
        }

        out
    }
}

/// Writes the circuit as OpenQASM 2.0 text into the provided path.
impl<P> LocalBuilder<P>
where
    P: Precision + ToPrimitive,
{
    /// Writes current circuit as OpenQASM 2.0 into `path`.
    pub fn write_openqasm_file(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let qasm = self.to_openqasm();
        let mut f = File::create(path)?;
        f.write_all(qasm.as_bytes())
    }
}

// --- helpers ---

fn emit_unitary<P: Precision + ToPrimitive>(
    u: &UnitaryMatrixObject<P>,
    indices: &[usize],
    out: &mut String,
) {
    match u {
        // Single-qubit gates; if given many indices, emit per index.
        UnitaryMatrixObject::X => for_each_q(indices, out, "x"),
        UnitaryMatrixObject::Y => for_each_q(indices, out, "y"),
        UnitaryMatrixObject::Z => for_each_q(indices, out, "z"),
        UnitaryMatrixObject::H => for_each_q(indices, out, "h"),
        UnitaryMatrixObject::S => for_each_q(indices, out, "s"),
        UnitaryMatrixObject::T => for_each_q(indices, out, "t"),

        // Controlled NOT: first is control, remaining targets.
        UnitaryMatrixObject::CNOT => {
            if !indices.is_empty() {
                let c = indices[0];
                for &t in &indices[1..] {
                    writeln!(out, "cx q[{}],q[{}];", c, t).unwrap();
                }
            }
        }

        // SWAP: 2 => swap a,b; 2k => pairwise swap i with i+k.
        UnitaryMatrixObject::SWAP => match indices.len() {
            0 | 1 => {}
            2 => writeln!(out, "swap q[{}],q[{}];", indices[0], indices[1]).unwrap(),
            n if n % 2 == 0 => {
                let half = n / 2;
                for i in 0..half {
                    writeln!(out, "swap q[{}],q[{}];", indices[i], indices[i + half]).unwrap();
                }
            }
            _ => {
                writeln!(
                    out,
                    "// swap with odd arity {:?} not directly supported",
                    indices
                )
                .unwrap();
            }
        },

        // Rz(theta) in radians or symbolic pi-rational.
        UnitaryMatrixObject::Rz(theta) => {
            let ang = format_angle(theta);
            for &q in indices {
                writeln!(out, "rz({}) q[{}];", ang, q).unwrap();
            }
        }

        // Global phase: comment only in OQ2.
        UnitaryMatrixObject::GlobalPhase(theta) => {
            writeln!(
                out,
                "// global phase {} (ignored in OpenQASM 2.0)",
                format_angle(theta)
            )
            .unwrap();
        }

        // Generic matrix: comment (not emitted).
        UnitaryMatrixObject::MAT(_) => {
            writeln!(
                out,
                "// generic unitary on {:?} (not emitted in OpenQASM 2.0)",
                indices
            )
            .unwrap();
        }
    }
}

fn for_each_q(indices: &[usize], out: &mut String, gate: &str) {
    for &q in indices {
        writeln!(out, "{} q[{}];", gate, q).unwrap();
    }
}

fn format_angle<P: Precision + ToPrimitive>(rot: &RotationObject<P>) -> String {
    match rot {
        RotationObject::Floating(p) => {
            // Decimal radians, trimmed.
            let f = p.to_f64().unwrap_or(0.0);
            format!("{:.12}", f)
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_string()
        }
        RotationObject::PiRational(r) => {
            // Emit as k*pi/m
            let normalized = normalize_ratio(r.clone());
            let numer = *normalized.numer();
            let denom = *normalized.denom();
            if denom == 1 {
                format!("{}*pi", numer)
            } else {
                format!("{}*pi/{}", numer, denom)
            }
        }
    }
}

fn normalize_ratio(r: Ratio<i64>) -> Ratio<i64> {
    // Ensure denominator > 0 for stable printing
    if *r.denom() < 0 {
        Ratio::new(-*r.numer(), -*r.denom())
    } else {
        r
    }
}

// =========================
//           TESTS
// =========================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::num::NonZeroUsize;

    // Bring builder traits (H, CNOT, RZ, measure, etc.)
    use crate::builder_traits::{CircuitBuilder, CliffordTBuilder, MeasurementBuilder, RotationsBuilder};

    // 1) Header + simple H + measure
    #[test]
    fn qasm_header_and_measure() {
        let mut b = LocalBuilder::<f64>::default();
        let q0 = b.register(NonZeroUsize::new(1).unwrap());

        // apply H on q0
        let h = b.make_h();
        let q0 = b.apply_circuit_object(q0, h).unwrap();

        // measure q0
        let (_q0, _mh) = b.measure(q0);

        let qasm = b.to_openqasm();
        assert!(qasm.contains("OPENQASM 2.0;"));
        assert!(qasm.contains("include \"qelib1.inc\";"));
        assert!(qasm.contains("qreg q[1];"));
        assert!(qasm.contains("creg c[1];"));
        assert!(qasm.contains("h q[0];"));
        assert!(qasm.contains("measure q[0] -> c[0];"));
    }

    // 2) CNOT produces cx, no classical register when no measurement
    #[test]
    fn qasm_cnot_no_creg() {
        let mut b = LocalBuilder::<f64>::default();
        let q0 = b.register(NonZeroUsize::new(1).unwrap());
        let q1 = b.register(NonZeroUsize::new(1).unwrap());
        let r = b.merge_two_registers(q0, q1);

        let cnot = b.make_cnot();
        let _r = b.apply_circuit_object(r, cnot).unwrap();

        let qasm = b.to_openqasm();
        assert!(qasm.contains("qreg q[2];"));
        assert!(qasm.contains("cx q[0],q[1];"));
        assert!(!qasm.contains("creg c[")); // no measurements -> no creg
    }

    // 3) Rz with PiRational emits symbolic angle
    #[test]
    fn qasm_rz_pi_rational() {
        let mut b = LocalBuilder::<f64>::default();
        let _q0 = b.register(NonZeroUsize::new(1).unwrap());
        let q1 = b.register(NonZeroUsize::new(1).unwrap());

        let _q1 = b.rz_pi_by(q1, 4).unwrap(); // pi/4

        let qasm = b.to_openqasm();
        assert!(qasm.contains("qreg q[2];"));
        assert!(qasm.contains("rz(1*pi/4) q[1];"));
    }

    // 4) Global phase is commented (ignored)
    #[test]
    fn qasm_global_phase_comment() {
        let mut b = LocalBuilder::<f64>::default();
        let q0 = b.register(NonZeroUsize::new(1).unwrap());
        let _q0 = b.apply_global_phase(q0, 0.3_f64); // arbitrary

        let qasm = b.to_openqasm();
        assert!(qasm.contains("// global phase"));
        assert!(qasm.contains("qreg q[1];"));
        // no creg
        assert!(!qasm.contains("creg c["));
    }

    // 5) Write file: ensures file exists and contains expected lines
    #[test]
    fn qasm_write_file_roundtrip() {
        let mut b = LocalBuilder::<f64>::default();
        let q0 = b.register(NonZeroUsize::new(1).unwrap());
        let q1 = b.register(NonZeroUsize::new(1).unwrap());
        let r = b.merge_two_registers(q0, q1);

        // H on q0, CX q0->q1, measure q0 and q1
        let h = b.make_h();
        let r = b.apply_circuit_object(r, h).unwrap();
        let cnot = b.make_cnot();
        let r = b.apply_circuit_object(r, cnot).unwrap();
        let (r, _m0) = b.measure(r);
        let (_r, _m1) = b.measure(r);

        // temp path
        let mut p = std::env::temp_dir();
        p.push("rustqip_test_export.qasm");

        b.write_openqasm_file(&p).unwrap();
        let text = fs::read_to_string(&p).unwrap();

        assert!(text.contains("OPENQASM 2.0;"));
        assert!(text.contains("qreg q[2];"));
        assert!(text.contains("creg c[2];"));
        assert!(text.contains("h q[0];"));
        assert!(text.contains("cx q[0],q[1];"));
        assert!(text.contains("measure q[0] -> c[0];"));
        assert!(text.contains("measure q[1] -> c[1];"));

        // best-effort cleanup
        let _ = fs::remove_file(p);
    }
}
