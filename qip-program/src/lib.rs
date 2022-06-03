extern crate proc_macro;

use proc_macro::{Delimiter, Group, TokenStream, TokenTree};
use std::iter::Peekable;
use std::str::FromStr;

/// Eats a ident <Group>
fn parse_register_and_indices<It: Iterator<Item=TokenTree>>(
    input_stream: &mut Peekable<It>,
) -> (String, Option<TokenTree>) {
    let register = if let Some(tokentree) = input_stream.next() {
        match tokentree {
            TokenTree::Ident(ident) => {
                ident.to_string()
            }
            _ => {
                panic!("Expected register identifier, found {:?}", tokentree)
            }
        }
    } else {
        panic!("Expected register identifier, found nothing")
    };

    let indices =
        if let Some(tokentree) = input_stream.peek() {
            match tokentree {
                TokenTree::Punct(p) if p.as_char() == ';' || p.as_char() == ',' => { None }
                _ => {
                    input_stream.next()
                }
            }
        } else {
            None
        };

    (register, indices)
}

/// Consumes a list of registers and punctuation up to the next line.
fn parse_list_of_registers<It: Iterator<Item=TokenTree>>(
    input_stream: &mut Peekable<It>,
) -> (Vec<Vec<String>>, Vec<Vec<Option<TokenTree>>>) {
    let mut register_groups: Vec<Vec<String>> = Vec::default();
    let mut index_groups: Vec<Vec<Option<TokenTree>>> = Vec::default();

    while let Some(p) = input_stream.peek() {
        match p {
            TokenTree::Group(_) => {
                // Group of registers coming up
                if let TokenTree::Group(group) = input_stream.next().unwrap() {
                    let mut it = group.stream().into_iter().peekable();
                    let (sub_register_groups, sub_index_groups) =
                        parse_list_of_registers(&mut it);
                    for v in &sub_register_groups {
                        if v.len() != 1 {
                            panic!("Register groups may not be nested");
                        }
                    }
                    let sub_register_groups = sub_register_groups
                        .into_iter()
                        .map(|mut v| v.pop().unwrap())
                        .collect();
                    let sub_index_groups = sub_index_groups
                        .into_iter()
                        .map(|mut v| v.pop().unwrap())
                        .collect();
                    register_groups.push(sub_register_groups);
                    index_groups.push(sub_index_groups);
                }
            }
            TokenTree::Punct(punct) if punct.as_char() == ',' => {
                input_stream.next();
            }
            TokenTree::Punct(punct) if punct.as_char() == ';' => {
                // Done with this line
                input_stream.next();
                break;
            }
            TokenTree::Ident(_) => {
                // A single register
                let (r, indices) = parse_register_and_indices(input_stream);
                register_groups.push(vec![r]);
                index_groups.push(vec![indices]);
            }
            p => panic!(
                "Expected group, identifier, comma, or semicolon, found {:?}",
                p
            ),
        }
    }

    (register_groups, index_groups)
}

#[proc_macro]
pub fn program(input_stream: TokenStream) -> TokenStream {
    let mut input_stream = input_stream.into_iter().peekable();
    let mut output_stream = TokenStream::new();

    // First we take the builder expression.
    let mut builder_stream = TokenStream::from_str("let _program_builder = ").unwrap();
    for tokentree in input_stream.by_ref() {
        let done = matches!(&tokentree, TokenTree::Punct(p) if p.as_char() == ';');
        builder_stream.extend(Some(tokentree));
        if done {
            break;
        }
    }
    output_stream.extend(builder_stream);

    // Parse input registers
    let mut input_registers = Vec::default();
    for tokentree in input_stream.by_ref() {
        match tokentree {
            TokenTree::Punct(p) if p.as_char() == ';' => {
                break;
            }
            TokenTree::Punct(p) if p.as_char() == ',' => {}
            TokenTree::Ident(ident) => input_registers.push(ident.to_string()),
            _ => panic!(
                "Expecting a register ident, a comma, or a semicolon, found {:?}",
                tokentree
            ),
        };
    }
    // println!("Input registers: {:?}", input_registers);

    let original_list = input_registers.clone();
    let original_size = original_list.len();
    input_registers.dedup();
    if original_size != input_registers.len() {
        panic!(
            "Input register list contained duplicates: {:?}",
            original_list
        );
    }

    for input_register in &input_registers {
        output_stream.extend(TokenStream::from_str(&format!("let mut {} = _program_builder.split_all_register({}).into_iter().map(|r| Some(r)).collect::<Vec<_>>();", input_register, input_register)).unwrap())
    }

    // Now we parse lines in the program.
    loop {
        let mut control = false;
        let mut control_bits = None;
        let mut function = String::new();
        let mut arguments = None;

        // Check if first word is 'control'
        if let Some(tokentree) = input_stream.next() {
            match tokentree {
                TokenTree::Ident(ident) if ident.to_string() == "control" => {
                    // println!("Found control");
                    control = true;
                }
                TokenTree::Ident(ident) => {
                    // println!("Found function: {}", ident.to_string());
                    function = ident.to_string();
                    if let Some(TokenTree::Group(g)) = input_stream.peek() {
                        if g.delimiter() == Delimiter::Parenthesis {
                            if let Some(TokenTree::Group(g)) = input_stream.next() {
                                // println!("Found arguments: {:?}", g);
                                arguments = Some(g.stream());
                            }
                        }
                    }
                }
                _ => {
                    panic!("Unexpected first token: {:?}", tokentree)
                }
            }
        }

        if control {
            let mut found_bit_group = false;
            // Now either we find control bits, or we find a function.
            if let Some(tokentree) = input_stream.next() {
                match tokentree {
                    TokenTree::Ident(ident) => {
                        // println!("Found function: {}", ident.to_string());
                        function = ident.to_string();
                        if let Some(TokenTree::Group(g)) = input_stream.peek() {
                            if g.delimiter() == Delimiter::Parenthesis {
                                if let Some(TokenTree::Group(g)) = input_stream.next() {
                                    // println!("Found arguments: {:?}", g);
                                    arguments = Some(g.stream());
                                }
                            }
                        }
                    }
                    TokenTree::Group(group) => {
                        // println!("Found control bits: {:?}", group);
                        found_bit_group = true;
                        control_bits = Some(group.stream());
                    }
                    _ => {
                        panic!("Unexpected token after `control`: {:?}", tokentree)
                    }
                }
            }
            if found_bit_group {
                // Now it had better be a function.
                if let Some(tokentree) = input_stream.next() {
                    match tokentree {
                        TokenTree::Ident(ident) => {
                            // println!("Found function: {}", ident.to_string());
                            function = ident.to_string();
                            if let Some(TokenTree::Group(g)) = input_stream.peek() {
                                if g.delimiter() == Delimiter::Parenthesis {
                                    if let Some(TokenTree::Group(g)) = input_stream.next() {
                                        // println!("Found arguments: {:?}", g);
                                        arguments = Some(g.stream());
                                    }
                                }
                            }
                        }
                        _ => {
                            panic!("Unexpected token after `control(bits)`: {:?}", tokentree)
                        }
                    }
                }
            }
        }

        // Now parse each of the register[indices]
        let (register_list, index_list) = parse_list_of_registers(&mut input_stream);
        // println!("Register groups: {:?}\tIndex groups: {:?}", register_list, index_list);

        let mut line_stream = TokenStream::new();

        // Pull relevant registers out.
        for (ri, (rs, is)) in register_list.iter().zip(index_list.iter()).enumerate() {
            let reg_name = format!("_program_register_{}", ri);

            let full_string = Some("None.into_iter()".to_string()).into_iter().chain(rs.iter().zip(is).map(|(r, s)| {
                if let Some(s) = s {
                    format!("qip::macros::program::QubitIndices::from({}).into_iter().map(|i| {}[i].take().unwrap())", s, r)
                } else {
                    format!("(0..{}.len()).map(|i| {}[i].take().unwrap())", r, r)
                }
            }).map(|s| format!(".chain({})", s))).collect::<String>();

            line_stream.extend(
                TokenStream::from_str(&format!(
                    "let {} = _program_builder.merge_registers({}).unwrap();",
                    reg_name, full_string
                ))
                    .unwrap(),
            );
        }
        output_stream.extend(line_stream);

        // Now use the registers to call the function.
        let mut start = 0;
        let mut builder_name = "_program_builder";
        let mut has_control_bits = false;
        if control {
            start = 1;

            if let Some(control_bits) = control_bits {
                output_stream.extend(TokenStream::from_str("let _control_bitmask = "));
                output_stream.extend(control_bits.clone());
                output_stream.extend(TokenStream::from_str(";"));
                output_stream.extend(TokenStream::from_str("let _program_register_0 = qip::macros::program::negate_bitmask(_program_builder, _program_register_0, _control_bitmask);"));
                has_control_bits = true;
            }

            output_stream.extend(TokenStream::from_str("let mut _control_program_builder = _program_builder.condition_with(_program_register_0);"));
            builder_name = "&mut _control_program_builder";
        }

        let args_string = if let Some(args) = arguments {
            format!("{},", args)
        } else {
            "".to_string()
        };

        let subsection = &register_list[start..];
        if subsection.len() == 1 {
            let register_name = format!("_program_register_{} ", start);
            let string = format!(
                "let {} = {}({}, {} {})?;",
                register_name, function, builder_name, args_string, register_name
            );
            // println!("Calling: {}", string);
            output_stream.extend(TokenStream::from_str(&string).unwrap());
        } else {
            let register_names = (start..register_list.len() - 1)
                .map(|i| format!("_program_register_{}, ", i))
                .chain(Some(format!(
                    "_program_register_{} ",
                    register_list.len() - 1
                )))
                .collect::<String>();
            let string = format!(
                "let ({}) = {}({}, {} {})?;",
                register_names, function, builder_name, args_string, register_names
            );
            // println!("Calling: {}", string);
            output_stream.extend(TokenStream::from_str(&string).unwrap());
        }

        // Now use the registers to call the function.
        if control {
            output_stream.extend(TokenStream::from_str(
                "let _program_register_0 = _control_program_builder.dissolve();",
            ));
            if has_control_bits {
                output_stream.extend(TokenStream::from_str("let _program_register_0 = qip::macros::program::negate_bitmask(_program_builder, _program_register_0, _control_bitmask);"));
            }
        }

        // Put registers back.
        let mut replace_qudits_stream = TokenStream::new();
        for (ri, (rs, is)) in register_list.iter().zip(index_list.iter()).enumerate() {
            let reg_name = format!("_program_register_{}", ri);
            replace_qudits_stream.extend(TokenStream::from_str(&format!("let mut {} = _program_builder.split_all_register({}).into_iter().map(|r| Some(r)).collect::<Vec<_>>(); let mut {}_index = 0;", reg_name, reg_name, reg_name)).unwrap());
            for (r, s) in rs.iter().zip(is.iter()) {
                let s = if let Some(s) = s {
                    format!(
                        "qip::macros::program::QubitIndices::from({})",
                        s
                    )
                } else {
                    format!("0..{}.len()", r)
                };

                replace_qudits_stream.extend(
                    TokenStream::from_str(&format!(
                        "for i in {} {{ {}[i] = {}[{}_index].take(); {}_index += 1;  }}",
                        s, r, reg_name, reg_name, reg_name
                    ))
                        .unwrap(),
                );
            }
        }
        output_stream.extend(replace_qudits_stream);

        if input_stream.peek().is_none() {
            break;
        }
    }

    // Return the registers
    for input_register in &input_registers {
        output_stream.extend(TokenStream::from_str(&format!("let {} = _program_builder.merge_registers({}.into_iter().flat_map(|r| r)).unwrap();", input_register, input_register)).unwrap())
    }

    let mut tuple_stream = TokenStream::new();

    if input_registers.len() == 1 {
        output_stream.extend(TokenStream::from_str(&format!(
            "Ok({})",
            input_registers[0]
        )));
    } else {
        for input_register in &input_registers {
            tuple_stream.extend(Some(
                TokenStream::from_str(&format!("{}, ", input_register)).unwrap(),
            ))
        }
        output_stream.extend(TokenStream::from_str(&format!(
            "Ok({})",
            TokenTree::Group(Group::new(Delimiter::Parenthesis, tuple_stream))
        )));
    }

    TokenStream::from(TokenTree::Group(proc_macro::Group::new(
        proc_macro::Delimiter::Brace,
        output_stream,
    )))
}
