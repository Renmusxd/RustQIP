mod qip;
mod operators;

fn apply_mats(builder: &operators::UnitaryBuilder, q: qip::Qubit) {

}

fn apply_circuit(builder: &operators::OpBuilder, q: qip::Qubit) {

}

fn main() {
    let mut session = qip::Session::new();
    let builder = operators::OpBuilder::new();

    let q1 = session.qubit(1);
    let q2 = session.qubit(3);
    let q3 = session.qubit(3);

    println!("Qs: {:#?}, {:#?}, {:#?}", q1, q2, q3);

    let q4 = qip::Qubit::merge(vec![q1, q2, q3]);

    println!("Qs: {:#?}", q4);
}
