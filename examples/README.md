# Examples

Each file in this folder:
- Is a demo of some quantum computing well known feature.
- It has documentation in the code (pending).
- Is a standalone file with a `main()` function that will be built into a standalone binary.
- When an example runs it will send useful output to the user.

## Build

Examples will be built by default when `cargo build` runs. They can also be built with:

```
cargo build --examples
```

or individually:

```
cargo build --example dense_coding
```

## Run

```
cargo run --example dense_coding
```
