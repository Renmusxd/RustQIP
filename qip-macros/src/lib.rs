extern crate proc_macro;

use std::collections::HashSet;
use std::fmt::Write;
use std::iter::Peekable;
use std::str::FromStr;

use proc_macro::{Delimiter, Spacing, TokenStream, TokenTree};

/// Eats a ident <Group>
fn parse_register_and_indices<It: Iterator<Item = TokenTree>>(
    input_stream: &mut Peekable<It>,
) -> (String, Option<TokenTree>) {
    let register = if let Some(tokentree) = input_stream.next() {
        match tokentree {
            TokenTree::Ident(ident) => ident.to_string(),
            _ => {
                panic!("Expected register identifier, found {:?}", tokentree)
            }
        }
    } else {
        panic!("Expected register identifier, found nothing")
    };

    let indices = if let Some(tokentree) = input_stream.peek() {
        match tokentree {
            TokenTree::Punct(p) if p.as_char() == ';' || p.as_char() == ',' => None,
            _ => input_stream.next(),
        }
    } else {
        None
    };

    (register, indices)
}

/// Consumes a list of registers and punctuation up to the next line.
fn parse_list_of_registers<It: Iterator<Item = TokenTree>>(
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
                    let (sub_register_groups, sub_index_groups) = parse_list_of_registers(&mut it);
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
                    control = true;
                }
                TokenTree::Ident(ident) => {
                    function = ident.to_string();
                    if let Some(TokenTree::Group(g)) = input_stream.peek() {
                        if g.delimiter() == Delimiter::Parenthesis {
                            if let Some(TokenTree::Group(g)) = input_stream.next() {
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
                        function = ident.to_string();
                        if let Some(TokenTree::Group(g)) = input_stream.peek() {
                            if g.delimiter() == Delimiter::Parenthesis {
                                if let Some(TokenTree::Group(g)) = input_stream.next() {
                                    arguments = Some(g.stream());
                                }
                            }
                        }
                    }
                    TokenTree::Group(group) => {
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
                            function = ident.to_string();
                            if let Some(TokenTree::Group(g)) = input_stream.peek() {
                                if g.delimiter() == Delimiter::Parenthesis {
                                    if let Some(TokenTree::Group(g)) = input_stream.next() {
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
                    format!("qip::macros::program::QubitIndices::from({})", s)
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
            TokenTree::Group(proc_macro::Group::new(Delimiter::Parenthesis, tuple_stream))
        )));
    }

    TokenStream::from(TokenTree::Group(proc_macro::Group::new(
        proc_macro::Delimiter::Brace,
        output_stream,
    )))
}

fn parse_function_args(arg_stream: TokenStream, to: &mut Vec<String>) {
    let mut arg_stream = arg_stream.into_iter().peekable();
    while let Some(token) = arg_stream.next() {
        match (token, arg_stream.peek()) {
            (TokenTree::Ident(ident), Some(TokenTree::Punct(punct)))
                if punct.as_char() == ':' && punct.spacing() == Spacing::Alone =>
            {
                to.push(ident.to_string());
            }
            _ => {}
        }
    }
}

#[proc_macro_attribute]
pub fn invert(attr: TokenStream, input_stream: TokenStream) -> TokenStream {
    // Output starts same as input.
    let mut output_stream = input_stream.clone();

    let mut attr = attr.into_iter().peekable();
    let new_function_name = attr.next();
    if let Some(TokenTree::Punct(_)) = attr.peek() {
        attr.next();
    }

    let mut non_register_args = Vec::default();
    while let Some(TokenTree::Ident(ident)) = attr.next() {
        non_register_args.push(ident.to_string());
        if let Some(TokenTree::Punct(_)) = attr.peek() {
            attr.next();
        }
    }

    let mut function_name = String::from("foo");
    let new_function_name = new_function_name.map(|s| s.to_string());

    let mut input_stream = input_stream.into_iter().peekable();
    // We will draw from the stream until we find the opening parens for the function arguments or generics.
    while let Some(token) = input_stream.next() {
        if let TokenTree::Ident(ident) = &token {
            match input_stream.peek() {
                Some(TokenTree::Group(group)) if group.delimiter() == Delimiter::Parenthesis => {
                    function_name = ident.to_string();

                    let to_add =
                        new_function_name.unwrap_or_else(|| format!("{}_inv", function_name));
                    let to_add = TokenStream::from_str(&to_add).unwrap();
                    output_stream.extend(to_add);

                    break;
                }
                Some(TokenTree::Punct(punct)) if punct.as_char() == '<' => {
                    function_name = ident.to_string();
                    let to_add =
                        new_function_name.unwrap_or_else(|| format!("{}_inv", function_name));
                    let to_add = TokenStream::from_str(&to_add).unwrap();
                    output_stream.extend(to_add);

                    break;
                }
                _ => {
                    let to_add = TokenStream::from(token);
                    output_stream.extend(to_add);
                }
            }
        } else {
            let to_add = TokenStream::from(token);
            output_stream.extend(to_add);
        }
    }

    // Now get the group with the function arguments.
    let mut function_args = vec![];
    for token in input_stream.by_ref() {
        let should_break = if let TokenTree::Group(group) = &token {
            if group.delimiter() == Delimiter::Parenthesis {
                parse_function_args(group.stream().clone(), &mut function_args);
                true
            } else {
                false
            }
        } else {
            false
        };
        let to_add = TokenStream::from(token);
        output_stream.extend(to_add);
        if should_break {
            break;
        }
    }

    // Now parse until curly braces, throw those out.
    for token in input_stream {
        match &token {
            TokenTree::Group(group) if group.delimiter() == Delimiter::Brace => {
                break;
            }
            _ => {
                let to_add = TokenStream::from(token);
                output_stream.extend(to_add);
            }
        }
    }

    let builder = function_args[0].clone();
    let new_builder = format!("_{builder}_new");

    let skip_args = HashSet::<String>::from_iter(non_register_args);

    let regs_only = function_args[1..]
        .iter()
        .filter_map(|s| {
            if !skip_args.contains(s) {
                Some(s.clone())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();

    let regs_list = regs_only.join(",");

    let regs_sizes = regs_only
        .iter()
        .map(|reg| format!("{reg}.n()"))
        .collect::<Vec<String>>()
        .join(",");

    let make_new_regs = regs_only.iter().fold(String::new(), |mut output, s| {
        let _ = write!(
            output,
            "let _{s}_new = {new_builder}.register({s}.n_nonzero());"
        );
        output
    });

    let new_regs_args =
        function_args[1..]
            .iter()
            .fold(format!("&mut {new_builder}"), |mut output, s| {
                let _ = if !skip_args.contains(s) {
                    write!(output, ", _{s}_new")
                } else {
                    write!(output, ", {s}")
                };
                output
            });

    let pop_regs = regs_only.iter().rev().fold(String::new(), |mut output, s| {
        let _ = write!(
            output,
            "let {s} = _selected_vec.pop().expect(&format!(\"Register {s} is missing!\"));"
        );
        output
    });

    let to_add = TokenStream::from(TokenTree::Group(proc_macro::Group::new(Delimiter::Brace, TokenStream::from_str(&format!("
        let _register_sizes = [{regs_sizes}];
        let mut {new_builder} = {builder}.new_similar();
        {make_new_regs}
        {function_name}({new_regs_args})?;
        let _subcircuit = {new_builder}.make_subcircuit()?;
        let _combined_r = {builder}.merge_registers([{regs_list}]).expect(\"Must have some registers.\");
        let _combined_r = {builder}.apply_inverted_subcircuit(_subcircuit, _combined_r)?;
        let mut _selected_vec = {builder}.split_relative_index_groups(_combined_r, _register_sizes.into_iter().scan(0, |acc, n| {{
            let range = *acc..*acc+n;
            *acc += n;
            Some(range)
        }})).get_all_selected().expect(\"All registers should have been selected\");
        {pop_regs}
        Ok(({regs_list}))
    ")).unwrap())));
    output_stream.extend(to_add);

    output_stream
}
