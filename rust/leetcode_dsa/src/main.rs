use leetcode_solutions::problems::*;

fn main() {
    // Test any problem here
    let result = p0001_two_sum::Solution::two_sum(vec![2, 7, 11, 15], 9);
    println!("Two Sum Result: {:?}", result);
    println!("Three Sum Result: {:?}", p0015_three_sum::Solution::three_sum(vec![-1, 0, 1, 2, -1, -4]));

}