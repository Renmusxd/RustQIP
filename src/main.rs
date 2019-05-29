mod qip;
use qip::qubits::*;
use qip::operators::*;
use qip::pipeline::run;


fn main() {
    let mut session = qip::qubits::Session::new();
    let mut builder = qip::operators::OpBuilder::new();

    let q1 = session.qubit(1);
    let q2 = session.qubit(3);
    let q3 = session.qubit(3);

    let q1 = builder.hadamard(q1);
    let mut c = builder.with_context(q1);
    let (q2, q3) = c.swap(q2, q3);
    let q1 = c.release_qubit();
    let q1 = builder.hadamard(q1);

    println!("Qs: {:?}, {:?}, {:?}", q1, q2, q3);

    let q4 = Qubit::merge(vec![q1, q2, q3]);

    println!("Qs: {:?}", q4);

//    run(&q4);
}
