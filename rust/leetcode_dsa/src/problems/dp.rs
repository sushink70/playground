use std::collections::HashMap;

fn fib_naive(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    fib_naive(n - 1) + fib_naive(n - 2)
}

fn fib_memo(n: u32, memo: &mut HashMap<u32, u64>) -> u64 {
    if let Some(&result) = memo.get(&n) {
        return result;
    }
    if n <= 1 {
        return n as u64;
    }
    let result = fib_memo(n - 1, memo) + fib_memo(n - 2, memo);
    memo.insert(n, result);
    result
}

fn fib_tab(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    let mut dp = vec![0u64; (n + 1) as usize];
    dp[1] = 1;
    for i in 2..=n as usize {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    dp[n as usize]
}

fn fib_optimized(n: u32) -> u64 {
    if n <= 1 {
        return n as u64;
    }
    let (mut prev2, mut prev1) = (0u64, 1u64);
    for _ in 2..n {
        let current = prev1 + prev2;
        prev2 = prev1;
        prev1 = current;
    }
    prev1
}