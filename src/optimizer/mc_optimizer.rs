use crate::errors::{CircuitError, CircuitResult};
use crate::optimizer::index_trie::IndexTrie;
use rand::prelude::SliceRandom;
use rand::Rng;
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::swap;
use std::path::Path;

#[derive(Debug)]
struct Node<CO> {
    indices: Vec<usize>,
    prev: Option<usize>,
    next: Option<usize>,
    co: CO,
}

#[derive(Debug)]
pub struct MonteCarloOptimizer<CO>
where
    CO: Eq + Hash,
{
    data: Vec<Node<CO>>,
    head: Option<usize>,
    trie: Option<IndexTrie<(Vec<usize>, CO), Vec<(Vec<usize>, CO)>>>,
    // nvars: usize,
}

impl<CO> MonteCarloOptimizer<CO>
where
    CO: Eq + Hash + Clone + Debug,
{
    fn verify(&self) -> bool {
        let mut prev = None;
        let mut sel = self.head;
        let mut count = 0;
        while let Some(isel) = sel {
            count += 1;
            if self.data[isel].prev != prev {
                println!(
                    "Verify failure on self.data[{}]: {:?}",
                    isel, &self.data[isel]
                );
                println!("{:?} != {:?}", self.data[isel].prev, prev);
                return false;
            }

            prev = sel;
            sel = self.data[isel].next;
        }
        count == self.data.len()
    }

    pub fn new_from_path<SC, F, P>(sc: SC, trie_path: P, f: F) -> CircuitResult<Self>
    where
        SC: IntoIterator<Item = (Vec<usize>, CO)>,
        F: Fn(&str) -> CircuitResult<CO>,
        P: AsRef<Path>,
    {
        let trie = IndexTrie::new_from_filepath(trie_path, f)?;
        Ok(Self::new(sc, trie))
    }

    pub fn new<SC: IntoIterator<Item = (Vec<usize>, CO)>>(
        sc: SC,
        replacement_trie: IndexTrie<(Vec<usize>, CO), Vec<(Vec<usize>, CO)>>,
    ) -> Self {
        let mut data = sc
            .into_iter()
            .map(|(indices, co)| Node {
                indices,
                prev: None,
                next: None,
                co,
            })
            .collect::<Vec<Node<CO>>>();
        let n = data.len();
        let head = if n > 0 {
            data.iter_mut().enumerate().for_each(|(index, node)| {
                node.prev = match index {
                    index if index == 0 => None,
                    index => Some(index - 1),
                };
                node.next = match index {
                    index if index == n - 1 => None,
                    index => Some(index + 1),
                };
            });
            Some(0)
        } else {
            None
        };
        // let nvars = data
        //     .iter()
        //     .map(|node| node.indices.iter().cloned().max())
        //     .flatten()
        //     .max()
        //     .unwrap_or(0);
        Self {
            data,
            head,
            trie: Some(replacement_trie),
            // nvars,
        }
    }

    /// Returns the index just before the replaced region, and the index just after.
    fn replace<It>(&mut self, start_inc: usize, end_inc: usize, with: It) -> CircuitResult<()>
    where
        It: IntoIterator<Item = (Vec<usize>, CO)>,
    {
        let mut sel = start_inc;
        let mut replacements = with.into_iter();
        loop {
            let replace = replacements.next();
            if let Some((indices, co)) = replace {
                self.data[sel].co = co;
                self.data[sel].indices = indices;
                if sel != end_inc {
                    sel = self.data[sel]
                        .next
                        .ok_or_else(|| CircuitError::new("Gave a non-closed range."))?;
                } else {
                    break;
                }
            } else {
                // Find number of required deletions.
                // end_inc still in same place since we have only done replacements so far.
                let mut num_rep = 1;
                let mut tmp_sel = sel;
                while tmp_sel != end_inc {
                    num_rep += 1;
                    tmp_sel = self.data[tmp_sel].next.unwrap();
                }
                // Perform deletions.
                for _ in 0..num_rep {
                    self.remove(sel);
                }

                break;
            }
        }
        replacements.fold(end_inc, |pos, (indices, co)| {
            self.insert_after(pos, indices, co)
        });

        assert!(self.verify());
        Ok(())
    }

    fn insert_after<V>(&mut self, after: usize, indices: V, co: CO) -> usize
    where
        V: Into<Vec<usize>>,
    {
        let next_node = self.data[after].next;
        let n = Node {
            indices: indices.into(),
            prev: Some(after),
            next: next_node,
            co,
        };
        let pos = self.data.len();
        self.data.push(n);

        self.data[after].next = Some(pos);
        if let Some(next) = next_node {
            self.data[next].prev = Some(pos);
        }

        assert!(self.verify());
        pos
    }

    fn remove(&mut self, pos: usize) -> Option<(Vec<usize>, CO)> {
        let last_pos = self.data.len() - 1;
        if pos < last_pos {
            self.swap_physical(pos, last_pos);
        }
        // Untie it
        if let Some(prev) = self.data[last_pos].prev {
            self.data[prev].next = self.data[last_pos].next;
        } else {
            self.head = self.data[last_pos].next;
        }
        if let Some(next) = self.data[last_pos].next {
            self.data[next].prev = self.data[last_pos].prev;
        }

        let tmp = self.data.pop().map(|n| (n.indices, n.co));
        assert!(self.verify());
        tmp
    }

    /// Swaps positions in memory but not order in linked list
    fn swap_physical(&mut self, a: usize, b: usize) {
        let a_ref = &self.data[a];
        let b_ref = &self.data[b];
        // println!("\nBefore: {}:{:?}\t{}:{:?}", a, a_ref, b, b_ref);
        let (a_prev, a_next) = (a_ref.prev, a_ref.next);
        let (b_prev, b_next) = (b_ref.prev, b_ref.next);

        if let Some(prev_a) = a_prev {
            self.data[prev_a].next = Some(b);
        } else {
            self.head = Some(b)
        }
        if let Some(next_a) = a_next {
            self.data[next_a].prev = Some(b);
        }
        if let Some(prev_b) = b_prev {
            self.data[prev_b].next = Some(a);
        } else {
            self.head = Some(a)
        }
        if let Some(next_b) = b_next {
            self.data[next_b].prev = Some(a);
        }

        if a_next == Some(b) {
            // a_prev -> a -> b -> b_next
            assert_eq!(b_prev, Some(a));
            self.data[a].prev = a_prev;
            self.data[a].next = Some(a);
            self.data[b].prev = Some(b);
            self.data[b].next = b_next;
        } else if b_next == Some(a) {
            // b_prev -> b -> a -> a_next
            assert_eq!(a_prev, Some(b));
            self.data[b].prev = b_prev;
            self.data[b].next = Some(b);
            self.data[a].prev = Some(a);
            self.data[a].next = a_next;
        }
        self.data.swap(a, b);
        assert!(self.verify());
    }

    /// Swap positions in linked list.
    fn swap_nodes(&mut self, a: usize, b: usize) {
        let (head, tail) = match (a, b) {
            (a, b) if a < b => {
                let (head, tail) = self.data.split_at_mut(b);
                (&mut head[a], &mut tail[0])
            }
            (a, b) if b < a => {
                let (head, tail) = self.data.split_at_mut(a);
                (&mut head[b], &mut tail[0])
            }
            _ => return,
        };
        swap(&mut head.co, &mut tail.co);
        swap(&mut head.indices, &mut tail.indices);
        assert!(self.verify());
    }

    fn sort_by_index(&mut self) {
        if let Some(mut sel) = self.head {
            let mut i = 0;
            while self.data[sel].next.is_some() {
                let a = &self.data[sel];
                let next_sel = a.next.unwrap();
                let b = &self.data[next_sel];
                let overlap = a.indices.iter().any(|i| b.indices.iter().any(|j| i == j));
                let should_swap = if overlap {
                    false
                } else {
                    // Since no overlap, just compare minima
                    let min_a = a.indices.iter().cloned().min().unwrap();
                    let min_b = b.indices.iter().cloned().min().unwrap();
                    min_a > min_b
                };
                if should_swap {
                    let a_index = sel;
                    let b_index = next_sel;
                    if let Some(a_prev) = a.prev {
                        sel = a_prev;
                    } else {
                        sel = next_sel;
                    }
                    self.swap_nodes(a_index, b_index);
                } else {
                    sel = next_sel
                }
                i += 1;
                if i > 10 {
                    break;
                }
            }
        }
        assert!(self.verify());
    }

    pub fn get_opts_depth(&self) -> usize {
        self.data.len()
    }

    pub fn get_ops(&self) -> Vec<(Vec<usize>, CO)> {
        let mut res = vec![];

        if let Some(mut sel) = self.head {
            while self.data[sel].next.is_some() {
                let a = &self.data[sel];
                res.push((a.indices.clone(), a.co.clone()));

                let next_sel = a.next.unwrap();
                sel = next_sel;
            }
            let a = &self.data[sel];
            res.push((a.indices.clone(), a.co.clone()));
        }
        res
    }

    fn insert_identities<F, R: Rng>(&mut self, beta: f64, energy_function: F, rng: &mut R)
    where
        F: Fn(&CO) -> i32,
    {
        let trie = self.trie.take().unwrap();

        // First lets put in some identities.
        let identities = trie.get_root_values();
        if !identities.is_empty() {
            let mut sel = self.head;
            while let Some(isel) = sel {
                let attempt = &identities[rng.gen_range(0..identities.len())];
                let energy: i32 = attempt.iter().map(|(_, co)| energy_function(co)).sum();
                let p = (-beta * (energy as f64)).exp();
                if rng.gen_bool(p) {
                    let mut base_indices = self.data[isel].indices.clone();
                    if let Some(prev) = self.data[isel].prev {
                        self.data[prev].indices.iter().for_each(|index| {
                            if !base_indices.contains(index) {
                                base_indices.push(*index)
                            }
                        })
                    }
                    if let Some(next) = self.data[isel].next {
                        self.data[next].indices.iter().for_each(|index| {
                            if !base_indices.contains(index) {
                                base_indices.push(*index)
                            }
                        })
                    }
                    let max_index_needed = attempt
                        .iter()
                        .map(|(indices, _)| indices.iter())
                        .flatten()
                        .cloned()
                        .max()
                        .unwrap();
                    if max_index_needed < base_indices.len() {
                        let last_index = attempt.iter().fold(isel, |index, (indices, co)| {
                            let mut indices = indices.clone();
                            indices.iter_mut().for_each(|i| {
                                *i = base_indices[*i];
                            });
                            self.insert_after(index, indices, co.clone())
                        });
                        sel = self.data[last_index].next;
                    } else {
                        sel = self.data[isel].next;
                    }
                } else {
                    sel = self.data[isel].next;
                }
            }
        }
        self.trie = Some(trie);
    }

    pub fn run_optimizer_pass<F, R: Rng>(
        &mut self,
        beta: f64,
        energy_function: F,
        rng: &mut R,
    ) -> CircuitResult<()>
    where
        F: Fn(&CO) -> i32,
    {
        // Sort by index to group similar operators.
        self.sort_by_index();

        // Insert some identities if possible.
        self.insert_identities(beta, &energy_function, rng);

        let trie = self.trie.take().unwrap();
        // Now lets go through all other identities
        let mut sel = self.head;
        let mut counter = 0;
        let res = loop {
            // Each time changes are made to the linked list, need to make a new walker since
            // previous entries are out of date.
            let mut walker_opt = Some(trie.index_walker());
            'outer: while let Some(isel) = sel {
                // Get the owned walker instance, ready to drop in order to make changes.
                let mut walker = walker_opt.take().unwrap();

                // Find any possible replacements for this node series.
                let current_node = &self.data[isel];
                assert_eq!(
                    current_node.indices,
                    {
                        let mut ni = current_node.indices.clone();
                        ni.dedup();
                        ni
                    },
                    "Duplicate indices in node: {:?}",
                    current_node
                );
                let mut res = walker
                    .feed_acc(
                        current_node.indices.clone(),
                        current_node.co.clone(),
                        (current_node, 0, 0),
                        |(node, energy, count), co| (node, energy + energy_function(co), count + 1),
                    )
                    .into_iter()
                    .map(|((starting_node, energy, count), indices, exchanges)| {
                        exchanges.into_iter().map(move |exchange| {
                            (starting_node, energy, count, indices.clone(), exchange)
                        })
                    })
                    .flatten()
                    .collect::<Vec<_>>();
                res.shuffle(rng);

                // Go through possible replacements.
                for (starting_node, energy, count, indices, mut exchange) in res {
                    let replacement_energy: i32 =
                        exchange.iter().map(|(_, co)| energy_function(co)).sum();
                    let prob = if replacement_energy < energy {
                        1.0
                    } else {
                        (beta * (energy - replacement_energy) as f64).exp()
                    };
                    let go = if prob < 1.0 { rng.gen_bool(prob) } else { true };
                    // If one catches, then make edits and drop the walker by breaking the
                    // outer loop.
                    if go {
                        exchange.iter_mut().for_each(|(op_indices, ..)| {
                            op_indices
                                .iter_mut()
                                .enumerate()
                                .for_each(|(i, x)| *x = indices[i])
                        });
                        let starting_index = self.get_index_of_node(starting_node);
                        let ending_index = isel;
                        counter -= count - exchange.len() as i64;
                        if counter < 0 {
                            counter = 0;
                        }
                        self.replace(starting_index, ending_index, exchange)?;

                        // Get the index of the op with the same count as the starting index.
                        let sindex = self.get_index_from_pos(counter as usize, None).unwrap();
                        sel = self.data[sindex].next;
                        counter += 1;
                        break 'outer;
                    }
                }
                // No changes, keep the current walker and move on.
                walker_opt = Some(walker);
                sel = self.data[isel].next;
                counter += 1;
            }
            if sel.is_none() {
                break Ok(());
            }
        };
        self.trie = Some(trie);
        res
    }

    fn get_index_of_node(&self, node: &Node<CO>) -> usize {
        if let Some(prev) = node.prev {
            self.data[prev].next.unwrap()
        } else {
            self.head.unwrap()
        }
    }

    fn get_index_from_pos(&self, pos: usize, hint: Option<(usize, usize)>) -> Option<usize> {
        if pos + 1 > self.data.len() {
            None
        } else {
            let (mut sel, hpos) = hint.unwrap_or((self.head.unwrap(), 0));
            let mut pos = pos - hpos;
            while pos > 0 {
                sel = self.data[sel].next.unwrap();
                pos -= 1;
            }
            Some(sel)
        }
    }
}

#[cfg(test)]
mod mc_opt_tests {
    use super::*;
    use rand::thread_rng;

    fn is_sorted<It, T>(it: It) -> bool
    where
        It: IntoIterator<Item = T>,
        T: Ord,
    {
        it.into_iter()
            .try_fold(None, |acc, i| match acc {
                None => Ok(Some(i)),
                Some(j) if i > j => Ok(Some(i)),
                _ => Err(()),
            })
            .is_ok()
    }

    #[test]
    fn basic_noop_sort() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 0), (vec![2, 3, 4], 1)],
            IndexTrie::default(),
        );
        mco.sort_by_index();
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn basic_sort() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 0), (vec![0, 4, 5], 1)],
            IndexTrie::default(),
        );
        mco.sort_by_index();
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(!is_sorted(ops));
        Ok(())
    }

    #[test]
    fn basic_sort_longer() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 1), (vec![6], 2), (vec![0, 4, 5], 0)],
            IndexTrie::default(),
        );
        mco.sort_by_index();
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn basic_remove() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![0], 0), (vec![1], 3), (vec![2], 1), (vec![3], 2)],
            IndexTrie::default(),
        );
        mco.remove(1);
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn adjacent_remove() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 0), (vec![6], 2), (vec![0, 4, 5], 1)],
            IndexTrie::default(),
        );
        mco.remove(1);
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn first_remove() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![0], 3), (vec![1], 0), (vec![2], 1), (vec![3], 2)],
            IndexTrie::default(),
        );
        mco.remove(0);
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn last_remove() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![0], 1), (vec![1], 2), (vec![2], 3), (vec![3], 0)],
            IndexTrie::default(),
        );
        mco.remove(3);
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn basic_replace() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 0), (vec![6], 3), (vec![0, 4, 5], 2)],
            IndexTrie::default(),
        );
        mco.replace(1, 1, [(vec![6], 1)])?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn overflow_replace() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![1, 2, 3], 0), (vec![6], 4), (vec![0, 4, 5], 3)],
            IndexTrie::default(),
        );
        mco.replace(1, 1, [(vec![6], 1), (vec![6], 2)])?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}", ops);
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn underflow_replace() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [
                (vec![1, 2, 3], 0),
                (vec![6], 2),
                (vec![6], 3),
                (vec![0, 4, 5], 1),
            ],
            IndexTrie::default(),
        );
        mco.replace(1, 2, [])?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn underflow_replace_multi() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [
                (vec![1, 2, 3], 0),
                (vec![6], 2),
                (vec![6], 3),
                (vec![0, 4, 5], 2),
            ],
            IndexTrie::default(),
        );
        mco.replace(1, 2, [(vec![6], 1)])?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn empty_replace() -> CircuitResult<()> {
        let mut mco = MonteCarloOptimizer::<usize>::new(
            [(vec![0], 2), (vec![1], 1), (vec![2], 0)],
            IndexTrie::default(),
        );
        mco.replace(0, 2, [])?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn simple_opt() -> CircuitResult<()> {
        let replacement_trie =
            IndexTrie::new_from_valid_lines(["4[1] 3[0] = 1[0]"], |s| s.parse().unwrap())?;
        println!("{:?}", replacement_trie);
        let mut mco =
            MonteCarloOptimizer::new([(vec![0], 3), (vec![1], 4), (vec![2], 2)], replacement_trie);
        let mut rng = thread_rng();
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn multi_opt() -> CircuitResult<()> {
        let replacement_trie =
            IndexTrie::new_from_valid_lines(["4[0] 3[0] = 0[0]", "6[0] 5[0] = 1[0]"], |s| {
                s.parse().unwrap()
            })?;
        println!("{:?}", replacement_trie);
        let mut mco = MonteCarloOptimizer::new(
            [
                (vec![0], 3),
                (vec![0], 4),
                (vec![1], 5),
                (vec![1], 6),
                (vec![2], 2),
            ],
            replacement_trie,
        );
        let mut rng = thread_rng();
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn multi_run_opt() -> CircuitResult<()> {
        let replacement_trie =
            IndexTrie::new_from_valid_lines(["4[0] 3[0] = 7[0]", "6[1] 7[0] = 1[0]"], |s| {
                s.parse().unwrap()
            })?;
        println!("{:?}", replacement_trie);
        let mut mco = MonteCarloOptimizer::new(
            [(vec![0], 3), (vec![0], 4), (vec![1], 6), (vec![2], 2)],
            replacement_trie,
        );
        let mut rng = thread_rng();
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn multi_run_opt_change_index() -> CircuitResult<()> {
        let replacement_trie =
            IndexTrie::new_from_valid_lines(["4[0] 3[0] = 7[0]", "6[1] 7[0] = 1[0]"], |s| {
                s.parse().unwrap()
            })?;
        println!("{:?}", replacement_trie);
        let mut mco = MonteCarloOptimizer::new(
            [(vec![3], 3), (vec![3], 4), (vec![4], 6), (vec![5], 2)],
            replacement_trie,
        );
        let mut rng = thread_rng();
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        mco.run_optimizer_pass(1000.0, |_| 1, &mut rng)?;
        println!("{:?}", mco.get_ops());
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert!(is_sorted(ops));
        Ok(())
    }

    #[test]
    fn identity_inf_temp() -> CircuitResult<()> {
        let replacement_trie =
            IndexTrie::new_from_valid_lines(["2[0] 1[0] = I"], |s| s.parse().unwrap())?;
        println!("{:?}", replacement_trie);
        let mut mco = MonteCarloOptimizer::new([(vec![0], 0)], replacement_trie);
        let mut rng = thread_rng();

        mco.insert_identities(0.0, |_| 1, &mut rng);
        let ops = mco.get_ops().into_iter().map(|(_, i)| i);
        let ops = ops.collect::<Vec<_>>();
        println!("{:?}, {}", ops, is_sorted(ops.clone()));
        assert_eq!(ops.len(), 3);
        assert!(is_sorted(ops));
        Ok(())
    }
}
