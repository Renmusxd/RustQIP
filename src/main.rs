mod qip;
use qip::qubits::*;
use qip::operators::*;

fn main() {
    let mut session = qip::qubits::Session::new();
    let builder = qip::operators::OpBuilder::new();

    let q1 = session.qubit(1);
    let q2 = session.qubit(3);
    let q3 = session.qubit(3);

    let mut c = builder.make_builder_with_context(q1).expect("Broken");
    let q2 = c.not(q2);
    let q1 = c.release_qubit();

    println!("Qs: {:#?}, {:#?}, {:#?}", q1, q2, q3);

    let q4 = Qubit::merge(vec![q1, q2, q3]);

    println!("Qs: {:#?}", q4);
}
