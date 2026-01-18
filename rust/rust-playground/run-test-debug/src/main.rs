fn take_ref(x: &i32) {}

fn main() {
    println!("Hello, world!");
    let b = Box::new(10);
    println!("{}", *b);
    let c = Box::new(5);
    take_ref(&b);
}
