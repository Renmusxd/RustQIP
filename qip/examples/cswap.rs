use qip::prelude::*;
use std::num::NonZeroUsize;


fn main() -> Result<(), CircuitError> {

    let mut b = LocalBuilder::<f64>::default();
    let n = NonZeroUsize::new(3).unwrap();

    let q = b.qubit();
    let ra = b.register(n);
    let rb = b.register(n);

    let q = b.h(q);

    let mut cb = b.condition_with(q);
    let (ra, rb) = cb.swap(ra, rb).unwrap();
    let q = cb.dissolve();

    let q = b.h(q);

    let (q, m_handle) = b.measure(q);


    let (_, measured) = b.calculate_state_with_init([(&ra, 0b000), (&rb, 0b001)]);
    let (result, p) = measured.get_measurement(m_handle);
    println!("Measured: {:?} (with chance {:?})", result, p);

    Ok(())

}