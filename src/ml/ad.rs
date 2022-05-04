use std::{
    cell::RefCell,
    collections::HashMap,
};

use super::ad_number::{
    ADNumber,
};

#[derive(Debug)]
struct PartialDerivative {
    with_respect_to_id: usize,
    differential: f32,
}

#[derive(Debug)]
struct TapeRecord {
    partials: Vec<PartialDerivative>
}

#[derive(Debug)]
struct Tape {
    records: Vec<TapeRecord>
}

type GradientsMap = HashMap<usize, Vec<f32>>;

pub struct AD {
    tapes: RefCell<HashMap<usize, Tape>>,
    max_tape_id: RefCell<usize>,
    gradients: RefCell<GradientsMap>,
}

impl PartialDerivative {
    pub fn wrt_id(&self) -> usize {
        self.with_respect_to_id
    }
}

impl Tape {
    fn new() -> Tape {
        Tape {
            records: Vec::new()
        }
    }
}

impl TapeRecord {
    fn new() -> TapeRecord {
        TapeRecord {
            partials: Vec::new()
        }
    }

    fn record(
        &mut self,
        dependency: &ADNumber,
        differential: f32,
    ) -> &mut Self {
        self.partials.push(PartialDerivative {
            with_respect_to_id: dependency.id(),
            differential,
        });
        self
    }
}

impl std::fmt::Debug for AD {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "AD {{ tapes: {:#?} }}", self.tapes.borrow())
    }
}

static mut AD_INSTANCE: Option<AD> = None;

impl <'a> AD {
    pub fn new() -> AD {
        AD {
            tapes: RefCell::new(vec![(1, Tape::new())].into_iter().collect()),
            max_tape_id: RefCell::new(0),
            gradients: RefCell::new(HashMap::new()),
        }
    }

    pub fn get_instance() -> &'a AD {
        unsafe {
            if AD_INSTANCE.is_none() {
                    AD_INSTANCE = Some(AD::new());
            }

            AD_INSTANCE.as_ref().unwrap()
        }
    }

    fn reset(&self) {
        self.max_tape_id.replace(0);
        self.gradients.replace(HashMap::new());
        self.tapes.replace(HashMap::new());
    }

    fn next_tape_id(&self) -> usize {
        let mut id = self.max_tape_id.borrow_mut();
        *id += 1;
        *id
    }

    pub fn current_tape_id(&self) -> usize {
        if *self.max_tape_id.borrow() == 0 {
            self.next_tape_id()
        } else {
            *self.max_tape_id.borrow()
        }
    }

    pub fn create_constant(&self, scalar: f32) -> ADNumber {
        ADNumber::new(0, 0, Some(self), scalar)
    }

    pub fn create_variable(&self, scalar: f32) -> ADNumber {
        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();

        let id = tape.records.len();
        tape.records.push(TapeRecord::new());

        ADNumber::new(tape_id, id, Some(self), scalar)
    }

    pub fn create_binary_composite(
        &self,
        left: &ADNumber,
        right: &ADNumber,
        result: f32,
        diff_wrt_left: f32,
        diff_wrt_right: f32,
    ) -> ADNumber {
        if left.is_constant() && right.is_constant() {
            return self.create_constant(result);
        }

        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).expect("to find the tape");

        let mut rec = TapeRecord::new();
        if left.is_variable() {
            rec.record(left, diff_wrt_left);
        }

        if right.is_variable() {
            rec.record(right, diff_wrt_right);
        }

        let id = tape.records.len();
        tape.records.push(rec);

        ADNumber::new(tape_id, id, Some(self), result)
    }

    pub fn create_unary_composite(
        &self,
        operand: &ADNumber,
        result: f32,
        diff_wrt_operand: f32,
    ) -> ADNumber {
        let tape_id = self.current_tape_id();
        let mut tapes = self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();
        let id = tape.records.len();

        if operand.is_constant() {
            return ADNumber::new(0, 0, Some(self), result);
        }

        let mut rec = TapeRecord::new();
        rec.record(
            operand, diff_wrt_operand,
        );
        tape.records.push(rec);

        ADNumber::new(tape_id, id, Some(self), result)
    }

    fn compute_gradient(&self, y: &ADNumber) -> Vec<f32> {
        let tape = &self.tapes.borrow()[&y.tape_id()];
        let mut dy = vec![0.0; y.id()+1];
        dy[y.id()] = 1.0;

        // - go through each computation backwards
        // - increment gradient for each variable the computation depends on
        for (i, record) in tape.records.iter().enumerate().rev() {
            for partial in &record.partials {
                dy[partial.with_respect_to_id] += partial.differential * dy[i];
            }
        }

        dy
    }

    pub fn diff(&self, y: &ADNumber, wrt: &ADNumber) -> f32 {
        let mut gradients = self.gradients.borrow_mut();
        match gradients.get(&y.id()) {
            Some(dy) => dy[wrt.id()],
            None => {
                let dy = self.compute_gradient(y);
                let res = dy[wrt.id()];
                gradients.insert(y.id(), dy);
                res
            }
        }
    }
}
