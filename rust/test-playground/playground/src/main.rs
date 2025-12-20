fn main() {
    // Your testing ground
    test_arrays();
    test_linked_lists();
    test_loops();
}

fn test_arrays() {
    println!("=== Array Operations ===");
    
    // Different ways to create arrays
    let arr1 = [1, 2, 3, 4, 5];
    let arr2: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut arr3 = Vec::with_capacity(10);
    
    // Access
    println!("First element: {}", arr1[0]);
    println!("Last element: {}", arr2[arr2.len() - 1]);
    
    // Iterate
    for (i, &val) in arr1.iter().enumerate() {
        println!("Index {}: {}", i, val);
    }
    
    // Mutable operations
    arr3.push(42);
    arr3.pop();
    
    println!();
}

fn test_linked_lists() {
    println!("=== Linked List (Vec simulation) ===");
    
    use std::collections::LinkedList;
    let mut list = LinkedList::new();
    
    list.push_back(1);
    list.push_front(0);
    
    for item in &list {
        println!("{}", item);
    }
    
    println!();
}

fn test_loops() {
    println!("=== Loop Patterns ===");
    
    // Range loops
    for i in 0..5 {
        print!("{} ", i);
    }
    println!();
    
    // While loop
    let mut x = 0;
    while x < 3 {
        print!("{} ", x);
        x += 1;
    }
    println!();
    
    // Loop with break
    let mut counter = 0;
    let result = loop {
        counter += 1;
        if counter == 10 {
            break counter * 2;
        }
    };
    println!("Result: {}", result);
    
    println!();
}
