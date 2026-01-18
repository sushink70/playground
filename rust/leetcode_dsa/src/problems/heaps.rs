use std::collections::BinaryHeap;
use std::cmp::Reverse;

struct MaxHeap<T> {
    heap: BinaryHeap<T>,
}

impl<T: Ord> MaxHeap<T> {
    fn new() -> Self {
        MaxHeap {
            heap: BinaryHeap::new(),
        }
    }

    fn push(&mut self, val: T) {
        self.heap.push(val);
    }

    fn pop(&mut self) -> Option<T> {
        self.heap.pop()
    }

    fn peek(&self) -> Option<&T> {
        self.heap.peek()
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}

struct MinHeap<T> {
        heap: BinaryHeap<Reverse<T>>
    }

impl <T: Ord> MinHeap<T>{
    fn new() -> Self {
        MinHeap {
            heap: BinaryHeap::new(),
        }
    }

    fn push(&mut self, val: T) {
        self.heap.push(Reverse(val));
    }

    fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|Reverse(val)| val)
    }

    fn len(&self) -> usize {
        self.heap.len()
    }
}