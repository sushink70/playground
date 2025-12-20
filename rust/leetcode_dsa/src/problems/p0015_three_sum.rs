pub struct Solution;

impl Solution {
    pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        // Sort the numbers and use the two-pointer technique to find unique triplets that sum to zero
        nums.sort();
        let n = nums.len();
        let mut res: Vec<Vec<i32>> = Vec::new();

        for i in 0..n {

            // Skip duplicate values for the first element
            if i > 0 && nums[i] == nums[i - 1] {
                continue;
            }
            let mut left = i + 1;
            let mut right = n.saturating_sub(1);
            let target = -nums[i];

            while left < right {
                let sum = nums[left] + nums[right];
                if sum == target {
                    res.push(vec![nums[i], nums[left], nums[right]]);
                    left += 1;
                    // skip duplicates for left
                    while left < right && nums[left] == nums[left - 1] {
                        left += 1;
                    }
                    // move right inward and skip duplicates
                    if right == 0 { break; } // defensive, though right>left ensures right>0
                    right -= 1;
                    while left < right && nums[right] == nums[right + 1] {
                        if right == 0 { break; }
                        right = right.saturating_sub(1);
                    }
                } else if sum < target {
                    left += 1;
                } else {
                    if right == 0 { break; }
                    right -= 1;
                }
            }
        }

        res
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