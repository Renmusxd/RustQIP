use crate::errors::{CircuitError, CircuitResult};
use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::hash::Hash;
use std::io;
use std::io::BufRead;
use std::num::ParseIntError;
use std::path::Path;
use std::str::FromStr;

#[derive(Debug, Clone)]
struct Node<T, V> {
    values: Vec<V>,
    children: HashMap<T, Node<T, V>>,
}
impl<T, V> Default for Node<T, V> {
    fn default() -> Self {
        Self {
            values: vec![],
            children: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct IndexTrie<T, V> {
    data: Node<T, V>,
}

impl<T, V> Default for IndexTrie<T, V> {
    fn default() -> Self {
        Self {
            data: Node::default(),
        }
    }
}

impl<T, V> IndexTrie<T, V>
where
    T: Hash + Eq,
{
    pub fn new<It, ItIt>(data: It) -> Self
    where
        It: IntoIterator<Item = (ItIt, V)>,
        ItIt: IntoIterator<Item = T>,
    {
        let mut trie = Self::default();
        data.into_iter().for_each(|(s, value): (ItIt, V)| {
            let node = s.into_iter().fold(&mut trie.data, |acc, t| {
                let entry = acc.children.entry(t);
                entry.or_insert_with(Node::default)
            });
            node.values.push(value);
        });

        trie
    }
    pub fn get_root_values(&self) -> &[V] {
        &self.data.values
    }
}

impl<CO> IndexTrie<(Vec<usize>, CO), Vec<(Vec<usize>, CO)>>
where
    CO: Hash + Eq + Clone,
{
    pub fn new_from_valid_lines<It, F, S>(lines: It, f: F) -> CircuitResult<Self>
    where
        S: AsRef<str>,
        It: IntoIterator<Item = S>,
        F: Fn(&str) -> CO,
    {
        Self::new_from_lines(lines, |x| Ok(f(x)))
    }

    pub fn new_from_lines<It, F, S>(lines: It, f: F) -> CircuitResult<Self>
    where
        S: AsRef<str>,
        It: IntoIterator<Item = S>,
        F: Fn(&str) -> CircuitResult<CO>,
    {
        let mut lhss = vec![];
        let mut rhss = vec![];
        lines.into_iter().try_for_each(|line| {
            let line = line.as_ref();
            let line = if let Some((line, _comment)) = line.split_once("//") {
                line.trim()
            } else {
                line.trim()
            };
            if !line.is_empty() {
                let (lhs, rhs) = line
                    .split_once("=")
                    .ok_or_else(|| CircuitError::new("Line missing '='"))?;
                let lhs = parse_str(lhs, &f)?;
                let rhs = parse_str(rhs, &f)?;
                lhss.push(lhs);
                rhss.push(rhs);
            }
            Ok(())
        })?;
        let iter = lhss
            .into_iter()
            .zip(rhss.into_iter())
            .map(|(mut lhs, mut rhs)| {
                lhs.reverse();
                rhs.reverse();
                (lhs, rhs)
            })
            .map(|(lhs, rhs)| [(lhs.clone(), rhs.clone()), (rhs, lhs)])
            .flatten();

        Ok(Self::new(iter))
    }

    pub fn new_from_filepath<P, F>(filename: P, f: F) -> CircuitResult<Self>
    where
        P: AsRef<Path>,
        F: Fn(&str) -> CircuitResult<CO>,
    {
        let lines = read_lines(filename)
            .map_err(|err| CircuitError::new(format!("IOError: {:?}", err)))?
            .try_fold(vec![], |mut acc, line| {
                let line = line.map_err(|err| CircuitError::new(format!("IOError: {:?}", err)))?;
                acc.push(line);
                Ok(acc)
            })?;
        Self::new_from_lines(lines, f)
    }
}

// The output is wrapped in a Result to allow matching on errors
// Returns an Iterator to the Reader of the lines of the file.
fn read_lines<P>(filename: P) -> io::Result<io::Lines<io::BufReader<File>>>
where
    P: AsRef<Path>,
{
    let file = File::open(filename)?;
    Ok(io::BufReader::new(file).lines())
}

fn parse_num_array(s: &str) -> Result<Vec<usize>, ParseIntError> {
    s.split(',')
        .map(|s| s.trim())
        .try_fold(vec![], |mut acc, s| {
            let n = usize::from_str(s)?;
            acc.push(n);
            Ok(acc)
        })
}

fn parse_str<CO, F>(mut s: &str, f: F) -> Result<Vec<(Vec<usize>, CO)>, CircuitError>
where
    F: Fn(&str) -> CircuitResult<CO>,
{
    let mut res = vec![];
    // Expect ident[indices]
    while !s.is_empty() {
        if let Some(start_index) = s.find('[') {
            let co = f(s[0..start_index].trim())?;
            let end_index = s
                .find(']')
                .ok_or_else(|| CircuitError::new(format!("Error parsing string: {}", &s)))?;
            let num_array = parse_num_array(&s[start_index + 1..end_index]).map_err(|err| {
                CircuitError::new(format!(
                    "Error parsing string: {} -- {:?}",
                    &s[0..end_index + 1],
                    err
                ))
            })?;
            res.push((num_array, co));
            s = &s[end_index + 1..];
        } else {
            break;
        }
    }
    Ok(res)
}

impl<T, V> IndexTrie<(Vec<usize>, T), V>
where
    T: Hash + Eq + Clone,
    V: Clone,
{
    pub fn index_walker<K>(&self) -> IndexWalker<K, T, V>
    where
        K: Clone,
    {
        IndexWalker::new(self)
    }
}

#[derive(Debug)]
pub struct IndexWalker<'a, K, T, V> {
    trie: &'a IndexTrie<(Vec<usize>, T), V>,
    walkers: Option<Vec<(Vec<usize>, K, &'a Node<(Vec<usize>, T), V>)>>,
}

impl<'a, K, T, V> IndexWalker<'a, K, T, V>
where
    T: Hash + Eq + Clone,
    K: Clone,
    V: Clone,
{
    fn new(trie: &'a IndexTrie<(Vec<usize>, T), V>) -> Self {
        Self {
            trie,
            walkers: Some(vec![]),
        }
    }

    pub fn feed(&mut self, indices: Vec<usize>, t: T, marker: K) -> Vec<(K, Vec<usize>, Vec<V>)> {
        self.feed_acc(indices, t, marker, |k, _| k)
    }

    pub fn feed_acc<F>(
        &mut self,
        indices: Vec<usize>,
        t: T,
        init_state: K,
        f: F,
    ) -> Vec<(K, Vec<usize>, Vec<V>)>
    where
        F: Fn(K, &T) -> K,
    {
        assert_eq!(indices, {
            let mut ni = indices.clone();
            ni.dedup();
            ni
        });

        let mut new_walkers = vec![];
        let mut found_values = vec![];

        // Update all existing walkers.
        let walkers = self.walkers.take().unwrap();
        walkers.into_iter().for_each(|walker| {
            let mut new_t_indices = indices.clone();
            let (mut indices, state, node) = walker;
            let new_state = f(state, &t);
            let mut old_n = indices.len();
            new_t_indices.iter_mut().for_each(|i| {
                let existing = indices.iter().cloned().enumerate().find(|(_, ii)| ii == i);
                if let Some((loc, _)) = existing {
                    *i = loc;
                } else {
                    indices.push(*i);
                    *i = old_n;
                    old_n += 1;
                }
            });
            if let Some(node) = node.children.get(&(new_t_indices, t.clone())) {
                // Return found values.
                if !node.values.is_empty() {
                    found_values.push((new_state.clone(), indices.clone(), node.values.clone()));
                }
                // Make new walker
                new_walkers.push((indices, new_state, node));
            }
        });

        // First get a new one
        let first_indices = (0..indices.len()).collect();
        let new_state = f(init_state, &t);
        let new_node = self.trie.data.children.get(&(first_indices, t));
        if let Some(node) = new_node {
            // Return found values.
            if !node.values.is_empty() {
                found_values.push((new_state.clone(), indices.clone(), node.values.clone()));
            }
            // Make new walker
            new_walkers.push((indices, new_state, node));
        }
        self.walkers = Some(new_walkers);

        found_values
    }
}

#[cfg(test)]
mod index_trie_tests {
    use super::*;

    #[test]
    fn basic_test() {
        let a = (vec![0], 'a');
        let b = (vec![0], 'b');
        let c = (vec![0, 1], 'c');
        let d = (vec![1, 2], 'd');

        let trie = IndexTrie::new([
            (vec![a.clone(), b.clone()], 'A'),
            (vec![a.clone(), c.clone()], 'B'),
            (vec![d.clone(), a], 'C'),
            (vec![b, d.clone()], 'D'),
            (vec![c, d], 'E'),
        ]);
        let mut walker = trie.index_walker();

        assert_eq!(walker.feed(vec![0], 'a', '1'), vec![]);
        assert_eq!(
            walker.feed(vec![0], 'b', '2'),
            vec![('1', vec![0], vec!['A'])]
        );
        assert_eq!(
            walker.feed(vec![1, 2], 'd', '3'),
            vec![('2', vec![0, 1, 2], vec!['D'])]
        );
        assert_eq!(walker.feed(vec![3], 'a', '4'), vec![]);
        assert_eq!(
            walker.feed(vec![3, 0], 'c', '5'),
            vec![('4', vec![3, 0], vec!['B'])]
        );
        assert_eq!(
            walker.feed(vec![0, 1], 'd', '6'),
            vec![('5', vec![3, 0, 1], vec!['E'])]
        );
    }

    #[test]
    fn file_read_test() -> CircuitResult<()> {
        let trie =
            IndexTrie::new_from_valid_lines(["B[0,1]A[0,1] = C[0]", "A[0,1]A[1,0]=I"], |a| {
                a.to_string()
            })?;
        let mut walker = trie.index_walker();
        assert_eq!(walker.feed(vec![0, 1], "A".to_string(), '1'), vec![]);
        let res = walker.feed(vec![0, 1], "B".to_string(), '2');
        assert_eq!(
            res,
            vec![('1', vec![0, 1], vec![vec![(vec![0], "C".to_string())]])]
        );
        Ok(())
    }
}
