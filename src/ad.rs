use std::cell::RefCell;
use std::ops;
use std::collections::HashMap;

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

impl std::fmt::Debug for AD {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        writeln!(f, "AD {{ tapes: {:#?} }}", self.tapes.borrow())
    }
}

#[derive(Copy, Clone)]
pub struct ADNumber<'a> {
    ad: &'a AD,
    id: usize,
    tape_id: usize,
    scalar: f32,
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
        with_respect_to: &ADNumber,
        differential: f32,
    ) -> &mut Self {
        self.partials.push(PartialDerivative {
            with_respect_to_id: with_respect_to.id,
            differential,
        });
        self
    }
}

impl <'a> ADNumber<'a> {
    pub fn scalar(&self) -> f32 {
        self.scalar
    }

    // This is not differentiable
    pub fn is_constant(&self) -> bool {
        self.tape_id == 0
    }

    // This is differentiable
    pub fn is_variable(&self) -> bool {
        self.tape_id > 0
    }

    pub fn diff(&self, wrt: &ADNumber) -> f32 {
        self.ad.diff(self, wrt)
    }
}

impl AD {
    pub fn new() -> AD {
        let mut tapes = HashMap::new();
        tapes.insert(1, Tape::new());
        AD {
            tapes: RefCell::new(tapes),
            max_tape_id: RefCell::new(0),
            gradients: RefCell::new(HashMap::new()),
        }
    }

    fn next_tape_id(&self) -> usize {
        let mut max_id = self.max_tape_id.borrow_mut();
        *max_id += 1;
        *max_id
    }

    fn max_tape_id(&self) -> usize {
        if *self.max_tape_id.borrow() == 0 {
            self.next_tape_id()
        } else {
            *self.max_tape_id.borrow()
        }
    }

    fn get_tape_mut(&mut self) -> &mut Tape {
        let tape_id = *self.max_tape_id.borrow();
        let tape = self.tapes.get_mut().get_mut(&tape_id).unwrap();
        tape
    }

    pub fn create_constant(&self, scalar: f32) -> ADNumber {
        ADNumber {
            ad: self,
            id: 0,
            tape_id: 0,
            scalar,
        }
    }

    pub fn create_variable(&self, scalar: f32) -> ADNumber {
        let tape_id = self.max_tape_id();
        let mut tapes = self.tapes.borrow_mut();

        let value = ADNumber {
            ad: self,
            id: 0,
            tape_id,
            scalar,
        };

        let tape = tapes.get_mut(&tape_id).unwrap();
        tape.records.push(TapeRecord::new());

        value
    }

    pub fn create_binary_composite(
        &self,
        left: &ADNumber,
        right: &ADNumber,
        result: f32,
        diff_wrt_left: f32,
        diff_wrt_right: f32,
    ) -> ADNumber {
        let tape_id = self.max_tape_id();
        let tapes = &mut self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();

        if left.is_constant() && right.is_constant() {
            return self.create_constant(result);
        }

        let mut rec = TapeRecord::new();
        rec.record(
            &left, diff_wrt_left,
        ).record(
            &right, diff_wrt_right,
        );

        let id = tape.records.len();
        tape.records.push(rec);

        ADNumber {
            ad: self,
            id,
            tape_id,
            scalar: result,
        }
    }

    pub fn create_unary_composite(
        &self,
        operand: &ADNumber,
        result: f32,
        diff_wrt_operand: f32,
    ) -> ADNumber {
        let tape_id = self.max_tape_id();
        let tapes = &mut self.tapes.borrow_mut();
        let tape = tapes.get_mut(&tape_id).unwrap();
        let id = tape.records.len();

        if operand.is_constant() {
            return ADNumber {
                ad: self,
                id: 0,
                tape_id: 0,
                scalar: result,
            };
        }

        let mut rec = TapeRecord::new();
        rec.record(
            operand, diff_wrt_operand,
        );

        ADNumber {
            ad: self,
            id,
            tape_id,
            scalar: result,
        }
    }

    fn compute_gradient(&self, y: &ADNumber) -> Vec<f32> {
        let tape = &self.tapes.borrow()[&y.tape_id];
        let mut dy = vec![0.0; y.id+1];
        dy[y.id] = 1.0;

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
        match gradients.get(&y.id) {
            Some(dy) => dy[wrt.id],
            None => {
                let dy = self.compute_gradient(y);
                let res = dy[wrt.id];
                gradients.insert(y.id, dy);
                res
            }
        }
    }
}

impl <'a> ops::Add<ADNumber<'a>> for ADNumber<'a> {
    type Output = ADNumber<'a>;

    fn add(self, other: ADNumber<'a>) -> ADNumber {
        self.ad.create_binary_composite(
            &self, &other,
            self.scalar + other.scalar,
            1.0, 1.0,
        )
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let ad = AD::new();
        let x = ad.create_variable(2.0);
        let y = x + x;
        println!("{:#?}", ad);
        assert_eq!(y.scalar(), 4.0);
        assert_eq!(y.diff(&x), 2.0);
    }
}
