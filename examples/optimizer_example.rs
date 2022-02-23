use qip::macros::program_ops::*;
use qip::prelude::*;
use rand::{thread_rng, Rng};
use std::num::NonZeroUsize;

fn make_random_circuit<P: Precision, CB: CliffordTBuilder<P> + Conditionable, R: Rng>(
    cb: &mut CB,
    mut r: CB::Register,
    l: usize,
    rng: &mut R,
) -> Result<CB::Register, CircuitError> {
    const NGATES: usize = 7;
    let n = r.n();
    for _ in 0..l {
        let gatenum = rng.gen_range(0..NGATES);
        if gatenum == 0 {
            let qubit_numa = rng.gen_range(0..n);
            let qubit_numb = rng.gen_range(0..n - 1);
            let qubit_numb = if qubit_numb >= qubit_numa {
                qubit_numb + 1
            } else {
                qubit_numb
            };
            assert_ne!(qubit_numa, qubit_numb);
            r = program!(cb, r;
                control not r[qubit_numa], r[qubit_numb];
            )?;
        } else {
            let qubit_num = rng.gen_range(0..n);
            r = match gatenum {
                1 => program!(cb, r;
                    x r[qubit_num];
                ),
                2 => program!(cb, r;
                    y r[qubit_num];
                ),
                3 => program!(cb, r;
                    z r[qubit_num];
                ),
                4 => program!(cb, r;
                    h r[qubit_num];
                ),
                5 => program!(cb, r;
                    s r[qubit_num];
                ),
                6 => program!(cb, r;
                    t r[qubit_num];
                ),
                _ => unreachable!(),
            }?;
        }
    }
    Ok(r)
}

fn main() -> Result<(), CircuitError> {
    let mut b = LocalBuilder::<f64>::default();
    let r = b.register(NonZeroUsize::new(3).unwrap());

    let mut rng = thread_rng();

    let _ = make_random_circuit(&mut b, r, 100, &mut rng)?;
    println!("Depth: {}", b.pipeline_depth());
    println!("{:?}", b.make_subcircuit()?);

    let (state, _) = b.calculate_state();
    println!("{:?}", state);

    let mut opt = b.make_circuit_optimizer_from_file("rules.txt")?;

    for _ in 0..10 {
        opt.run_optimizer_pass(100., |_| 1, &mut rng)?;
    }
    println!("Depth: {}", opt.get_opts_depth());
    println!("{:?}", opt.get_ops());

    // for x in 1..100 {
    //     opt.run_optimizer_pass(2.0, |_| 1, &mut rng)?;
    //     // println!("{}\tDepth: {}", x, opt.get_opts_depth());
    // }
    // for x in 1..1000 {
    //     opt.run_optimizer_pass((x as f64).sqrt(), |_| 1, &mut rng)?;
    //     // println!("{}\tDepth: {}", x, opt.get_opts_depth());
    // }
    // println!("Depth: {}", opt.get_opts_depth());
    // println!("{:?}", opt.get_ops());

    let mut b = LocalBuilder::<f64>::default();
    let r = b.register(NonZeroUsize::new(3).unwrap());
    b.apply_optimizer_circuit(r, opt.get_ops())?;
    let (state, _) = b.calculate_state();
    println!("{:?}", state);

    Ok(())
}
