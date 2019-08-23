use crate::*;

wrap_fn!(pub x, UnitaryBuilder::x, r);

wrap_fn!(pub y, UnitaryBuilder::y, r);

wrap_fn!(pub z, UnitaryBuilder::z, r);

wrap_fn!(pub not, UnitaryBuilder::not, r);

wrap_fn!(pub swap, (UnitaryBuilder::swap), ra, rb);

wrap_fn!(pub h, UnitaryBuilder::hadamard, r);
