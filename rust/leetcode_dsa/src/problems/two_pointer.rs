fn two_sum_sorted(nums: &[i32], target: i32) -> Vec<usize> {
    let mut left = 0;
    let mut right = nums.len() - 1;

    while left < right {
        let current_sum = nums[left] + nums[right];

        if current_sum == target {
            return vec![left, right];
        } else if current_sum < target {
            left += 1;
        } else {
            right -= 1;
        }
    }
    vec![]
}