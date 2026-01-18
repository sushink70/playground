pub fn merge_sort<T: Ord + Clone + Copy>(arr: &mut [T]) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    
    let mid = len / 2;
    
    merge_sort(&mut arr[..mid]);
    merge_sort(&mut arr[mid..]);
    
    let mut aux = arr.to_vec();
    merge(&arr[..mid], &arr[mid..], &mut aux);
    arr.copy_from_slice(&aux);
}

fn merge<T: Ord + Clone>(left: &[T], right: &[T], result: &mut [T]) {
    let (mut i, mut j, mut k) = (0, 0, 0);
    
    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            result[k] = left[i].clone();
            i += 1;
        } else {
            result[k] = right[j].clone();
            j += 1;
        }
        k += 1;
    }
    
    while i < left.len() {
        result[k] = left[i].clone();
        i += 1;
        k += 1;
        
    }
    
    while j < right.len() {
        result[k] = right[j].clone();
        j += 1;
        k += 1;
    }
}