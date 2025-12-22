fn array_sum(arr: &[i32], index: usize) -> i32 {
    if index >= arr.len() {
        return 0;
    }
    let current = arr[index];
    let rest = array_sum(arr, index + 1);

    current + rest
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_array_sum() {
        let arr = [1, 2, 3, 4, 5];
        assert_eq!(array_sum(&arr, 0), 15);
        assert_eq!(array_sum(&arr, 2), 12);
        assert_eq!(array_sum(&arr, 5), 0);
    }
}