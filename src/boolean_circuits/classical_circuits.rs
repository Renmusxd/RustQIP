extern crate rand;
use std::rc::Rc;
use std::ops::{BitAnd, BitOr, BitXor};

/// The set of operations which can be performed on wires.
#[derive(Debug)]
pub enum WireOp {
    /// Wires to AND together
    And(Vec<Wires>),
    /// Wires to XOR together
    Or(Vec<Wires>),
    /// Wires to OR together
    Xor(Vec<Wires>),
    /// A not of the given wires
    Not(Box<Wires>),
    /// Copy the given relative indices from Wires
    Copy(Rc<Wires>, Option<Vec<u64>>),
    /// Concat the given wires
    Concat(Vec<Wires>)
}

/// A struct representing a set of bits.
#[derive(Debug)]
pub struct Wires {
    n: u64,
    parent: Option<WireOp>
}

impl Wires {
    /// Make a new set of `n` bits.
    pub fn new(n: u64) -> Self {
        Wires {
            n,
            parent: None
        }
    }

    /// Make a new set of `n` bits whose values come from `parent` in some way.
    pub fn new_with_parent(n: u64, parent: WireOp) -> Wires {
        Wires {
            n,
            parent: Some(parent)
        }
    }

    /// Split each bit out into its own `Wires` struct of `n=1`.
    pub fn split_all(self) -> Vec<Wires> {
        let shared_wire = Rc::new(self);
        (0..shared_wire.n).map(|index| {
            Wires::new_with_parent(1, WireOp::Copy(shared_wire.clone(), Some(vec![index])))
        }).collect()
    }

    /// Concat a vec of `Wires` into a single bundle.
    pub fn concat_all(wires: Vec<Wires>) -> Wires {
        Wires::new_with_parent(wires.iter().map(|w| w.n).sum(), WireOp::Concat(wires))
    }

    /// Branch wires so both carry the same values.
    pub fn branch(self) -> (Wires, Wires) {
        let shared_wire = Rc::new(self);
        let n = shared_wire.n;
        let new_wire = Wires::new_with_parent(n, WireOp::Copy(shared_wire.clone(), None));
        let wire = Wires::new_with_parent(n, WireOp::Copy(shared_wire, None));
        (wire, new_wire)
    }
}

fn match_wire_sizes(a: Wires, b: Wires) -> (Wires, Wires) {
    let (an, bn) = (a.n, b.n);
    if a.n < b.n {
        let a = Wires::concat_all(vec![a, Wires::new(bn - an)]);
        (a, b)
    } else {
        let b = Wires::concat_all(vec![b, Wires::new(an - bn)]);
        (a, b)
    }
}


impl BitAnd for Wires {
    type Output = Wires;
    fn bitand(self, rhs: Wires) -> Wires {
        let (a, b) = match_wire_sizes(self, rhs);
        Wires::new_with_parent(a.n, WireOp::And(vec![a, b]))
    }
}

impl BitOr for Wires {
    type Output = Wires;
    fn bitor(self, rhs: Wires) -> Wires {
        let (a, b) = match_wire_sizes(self, rhs);
        Wires::new_with_parent(a.n, WireOp::Or(vec![a, b]))
    }
}

impl BitXor for Wires {
    type Output = Wires;
    fn bitxor(self, rhs: Wires) -> Wires {
        let (a, b) = match_wire_sizes(self, rhs);
        Wires::new_with_parent(a.n, WireOp::Xor(vec![a, b]))
    }
}
