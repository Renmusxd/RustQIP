export CARGO_INCREMENTAL=0
export RUSTFLAGS="-Zprofile -Ccodegen-units=1 -Copt-level=0 -Clink-dead-code -Coverflow-checks=off -Cinstrument-coverage"
export LLVM_PROFILE_FILE="qip-%p-%m.profraw"

cargo clean
cargo +nightly test

grcov ./target/debug/ -s . -t html --llvm --branch --ignore-not-existing -o ./target/debug/coverage/
