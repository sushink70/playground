pub struct Solution;

impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        // Your solution here
        todo!() // This panics with "not yet implemented" - helpful while coding
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example1() {
        let nums = vec![-1, 0, 1, 2, -1, -4];
        let result = Solution::three_sum(nums);
        // Expected: [[-1,-1,2],[-1,0,1]]
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_example2() {
        let nums = vec![0, 1, 1];
        let result = Solution::three_sum(nums);
        assert_eq!(result, Vec::<Vec<i32>>::new());
    }

    #[test]
    fn test_example3() {
        let nums = vec![0, 0, 0];
        let result = Solution::three_sum(nums);
        assert_eq!(result, vec![vec![0, 0, 0]]);
    }
}