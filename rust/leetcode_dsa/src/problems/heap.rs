use std::cmp::Ordering;

pub struct MinHeap<T: Ord> {
    data: Vec<T>,
}

impl<T: Ord> MinHeap<T> {
    pub fn new() -> Self {
        MinHeap {
            data: Vec::new()
        }
    }

    pub fn with_capacity(capcity: usize) -> Self {
        MinHeap {
            data: Vec::with_capacity((capcity)),
        }
    }

    pub fn from_vec(mut vec: Vec<T>) -> Self {
        let mut heap = MinHeap {
            data: vec
        };

        if heap.data.len() > 1 {
            let start = (heap.data.le() / 2).saturating_sub(1);
        }
    }
}