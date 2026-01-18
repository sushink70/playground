use leetcode_solutions::problems::*;

fn main() {
    // Test any problem here
    let nums = vec![2, 7, 11, 15];
    let result = two_sum_sorted::Solution::two_sum_sorted(&nums, 9);
    println!("Two Sum Result: {:?}", result);
    let result = p0001_two_sum::Solution::two_sum(vec![2, 7, 11, 15], 9);
    println!("Two Sum Result: {:?}", result);
    
    println!("Three Sum Result: {:?}", p0015_three_sum::Solution::three_sum(vec![-1, 0, 1, 2, -1, -4]));
}

