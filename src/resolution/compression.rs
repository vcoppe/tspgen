use std::collections::HashMap;

use clustering::{kmeans, Elem};
use ddo::{Compression, Problem, Decision};
use osrm_client::Location;
use smallbitset::Set64;

use crate::instance::Instance;

use super::model::{TspModel, TspState};

struct Item<'a> {
    id: usize,
    tsp: &'a TspModel,
}

impl<'a> Elem for Item<'a> {
    fn dimensions(&self) -> usize {
        self.tsp.instance.destinations.len()
    }

    fn at(&self, i: usize) -> f64 {
        self.tsp.instance.distances[self.id][i] as f64
    }
}

pub struct TspCompression<'a> {
    pub problem: &'a TspModel,
    pub meta_problem: TspModel,
    pub membership: HashMap<isize, isize>,
}

impl<'a> TspCompression<'a> {
    pub fn new(problem: &'a TspModel, n_meta_destinations: usize) -> Self {
        let mut elems = vec![];
        for i in 0..problem.instance.destinations.len() {
            elems.push(Item {
                id: i,
                tsp: problem,
            });
        }
        let clustering = kmeans(n_meta_destinations, &elems, 1000);
        let destinations = vec![Location::new(0.0, 0.0); n_meta_destinations];
        let distances = Self::compute_meta_distances(problem, &clustering.membership, n_meta_destinations);

        let meta_problem = TspModel {
            instance: Instance {
                destinations,
                distances,
            },
        };

        let mut membership = HashMap::new();
        for (i, j) in clustering.membership.iter().enumerate() {
            membership.insert(i as isize, *j as isize);
        }

        TspCompression {
            problem,
            meta_problem,
            membership,
        }
    }

    fn compute_meta_distances(tsp: &TspModel, membership: &Vec<usize>, n_meta_destinations: usize) -> Vec<Vec<f32>> {
        let mut meta_distances = vec![vec![f32::MAX; n_meta_destinations]; n_meta_destinations];
        
        for a in 0..n_meta_destinations {
            for b in 0..n_meta_destinations {
                for i in (0..tsp.instance.destinations.len()).filter(|i| membership[*i] == a) {
                    for j in (0..tsp.instance.destinations.len()).filter(|j| membership[*j] == b) {
                        meta_distances[a][b] = meta_distances[a][b].min(tsp.instance.distances[i][j]);
                    }
                }
            }
        }

        meta_distances
    }
}

impl<'a> Compression for TspCompression<'a> {
    type State = TspState;

    fn get_compressed_problem(&self) -> &dyn Problem<State = Self::State> {
        &self.meta_problem
    }

    fn compress(&self, state: &TspState) -> TspState {
        let mut current = Set64::empty();
        for i in state.current.iter() {
            current.insert(*self.membership.get(&(i as isize)).unwrap() as u8);
        }

        let mut must_visit = Set64::empty();
        for i in state.must_visit.iter() {
            must_visit.insert(*self.membership.get(&(i as isize)).unwrap() as u8);
        }

        let might_visit = Set64::empty();

        let depth = self.meta_problem.instance.destinations.len() - must_visit.len();

        TspState {
            depth,
            current,
            must_visit,
            might_visit,
        }
    }

    fn decompress(&self, solution: &Vec<Decision>) -> Vec<Decision> {
        solution.clone()
    }
}