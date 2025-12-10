Great! Let me help you set up a proper structure for solving DSA problems in RustRover. Here's a recommended approach:

## Setup Steps

### 1. Initialize a Cargo Workspace

```bash
cd /home/iamdreamer/Documents/projects/playground/dsa/rust/leetcode
cargo init --name leetcode_solutions
```

### 2. Create a Structure for Multiple Problems

I'll show you two approaches - choose what works best:

**Approach A: Single binary with modules (Recommended for practice)**
- One `src/main.rs` that can run any problem
- Each problem in a separate module file
- Easy to test and switch between problems

**Approach B: Multiple binaries**
- Each problem as a separate binary
- Configure in `Cargo.toml`
- More isolated but slightly more setup

Let me create the recommended structure for you:

### 3. Recommended Project Structure

```
leetcode/
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── lib.rs
│   └── problems/
│       ├── mod.rs
│       ├── p0001_two_sum.rs
│       ├── p0002_add_two_numbers.rs
│       └── ...
└── tests/
    └── problem_tests.rs
```

### 4. Setup Files

**Cargo.toml:**
```toml
[package]
name = "leetcode_solutions"
version = "0.1.0"
edition = "2021"

[dependencies]

[dev-dependencies]
```

**src/lib.rs:**
```rust
pub mod problems;

// Common data structures you might need
#[derive(Debug, PartialEq, Eq)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

use std::rc::Rc;
use std::cell::RefCell;
```

**src/problems/mod.rs:**
```rust
pub mod p0001_two_sum;
// Add more as you solve them
```

**src/problems/p0001_two_sum.rs (Example):**
```rust
pub struct Solution;

impl Solution {
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        use std::collections::HashMap;
        let mut map = HashMap::new();
        
        for (i, &num) in nums.iter().enumerate() {
            let complement = target - num;
            if let Some(&j) = map.get(&complement) {
                return vec![j as i32, i as i32];
            }
            map.insert(num, i);
        }
        
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_example1() {
        assert_eq!(Solution::two_sum(vec![2, 7, 11, 15], 9), vec![0, 1]);
    }

    #[test]
    fn test_example2() {
        assert_eq!(Solution::two_sum(vec![3, 2, 4], 6), vec![1, 2]);
    }
}
```

**src/main.rs:**
```rust
use leetcode_solutions::problems::*;

fn main() {
    // Test any problem here
    let result = p0001_two_sum::Solution::two_sum(vec![2, 7, 11, 15], 9);
    println!("Two Sum Result: {:?}", result);
}
```

### 5. Quick Commands

```bash
# Run main
cargo run

# Run tests for all problems
cargo test

# Run tests for specific problem
cargo test p0001

# Run with optimizations (for benchmarking)
cargo run --release

# Check code without building
cargo check
```

### 6. Workflow for Each New Problem

1. Create new file: `src/problems/p0XXX_problem_name.rs`
2. Add module declaration in `src/problems/mod.rs`
3. Copy the template structure (Solution struct + impl)
4. Add test cases
5. Run `cargo test p0XXX` to verify

### 7. RustRover Tips

- Use `Ctrl+Shift+T` to quickly create/navigate to tests
- Use `Ctrl+R` to run current test
- Right-click on test function → "Run" to run single test
- Use clippy for hints: `cargo clippy`

### Quick Start Script

Want me to create a script that generates this structure automatically? Let me know and I can provide a shell script to set everything up!

**Which approach do you prefer, or shall I create the initial files for you?**

Perfect! It's working! 🎉

## How `cargo test p0001` Works

When you run `cargo test p0001`, Cargo does the following:

1. **Searches for test names/modules** containing "p0001" anywhere in the path
2. **Matches** your module `problems::p0001_two_sum::tests`
3. **Runs only those tests** that match the filter

### Test Filtering Examples

```bash
# Run all tests in p0001_two_sum module
cargo test p0001

# Run all tests in p0001_two_sum module (same as above)
cargo test p0001_two_sum

# Run specific test function
cargo test test_example1

# Run all tests containing "two_sum"
cargo test two_sum

# Run all tests in problems module
cargo test problems

# Run ALL tests
cargo test
```

### How the Matching Works

Your test path is: `problems::p0001_two_sum::tests::test_example1`

So these will ALL match it:
- `cargo test p0001` ✅ (matches module name)
- `cargo test two_sum` ✅ (matches module name)
- `cargo test p0001_two` ✅ (partial match)
- `cargo test test_example1` ✅ (matches test function)
- `cargo test example` ✅ (matches test function name)

### Useful Test Commands

```bash
# Show test output even for passing tests
cargo test p0001 -- --show-output

# Run tests with detailed output
cargo test p0001 -- --nocapture

# List all tests without running
cargo test -- --list

# Run tests in single thread (useful for debugging)
cargo test p0001 -- --test-threads=1
```

### Pro Tips for Your Workflow

**Quick test while coding:**
```bash
# Watch mode - auto-run tests on file change (requires cargo-watch)
cargo install cargo-watch
cargo watch -x "test p0001"
```

**Create an alias for convenience:**
```bash
# Add to your ~/.bashrc or ~/.zshrc
alias ct='cargo test'
alias ctp='cargo test --'  # For passing extra args
```

Then you can just do:
```bash
ct p0001
ct p0001 -- --nocapture
```

Your setup is working perfectly! Now you can:
1. Create a new problem file like `p0002_add_two_numbers.rs`
2. Add it to `problems/mod.rs`
3. Code your solution with tests
4. Run `cargo test p0002` to verify

Happy coding! 🦀

Great! Let me show you the complete workflow for solving a new problem.

## Step-by-Step Workflow for Solving Problems

### 1. Create the Problem File

```bash
# From your leetcode directory
touch src/problems/p0015_three_sum.rs
```

### 2. Add Module Declaration

Edit `src/problems/mod.rs`:
```rust
pub mod p0001_two_sum;
pub mod p0015_three_sum;  // Add this line
```

### 3. Start with Problem Template

In `src/problems/p0015_three_sum.rs`:

```rust
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
```

### 4. Development Workflow

**Option A: Watch Mode (Recommended for active coding)**

Open terminal in RustRover and run:
```bash
cargo watch -x "test p0015"
```

Now every time you save the file, tests auto-run! You'll see instant feedback.

**Option B: Manual Testing**

```bash
# Run tests for this problem
cargo test p0015

# Run with output visible
cargo test p0015 -- --nocapture

# Run specific test
cargo test test_example1
```

### 5. Writing Your Solution - Iterative Approach

**Step 1: Start Simple (Even if Wrong)**

```rust
impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        // Start with brute force to understand the problem
        // TODO: implement
        result
    }
}
```

**Step 2: Add Debug/Print Statements**

```rust
impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        println!("Input: {:?}", nums);
        
        let mut result = Vec::new();
        
        // Your logic here
        
        println!("Output: {:?}", result);
        result
    }
}
```

Run with: `cargo test p0015 -- --nocapture` to see prints

**Step 3: Implement Step by Step**

```rust
impl Solution {
    pub fn three_sum(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        let mut result = Vec::new();
        
        // Step 1: Sort (test this works)
        nums.sort_unstable();
        println!("Sorted: {:?}", nums);
        
        // Step 2: Add outer loop (test iteration)
        for i in 0..nums.len() {
            println!("i={}, nums[i]={}", i, nums[i]);
        }
        
        // Step 3: Continue building...
        
        result
    }
}
```

### 6. Testing Strategy

**Create Different Test Cases:**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let nums = vec![-1, 0, 1];
        let result = Solution::three_sum(nums);
        assert_eq!(result, vec![vec![-1, 0, 1]]);
    }

    #[test]
    fn test_empty() {
        let nums = vec![];
        let result = Solution::three_sum(nums);
        assert!(result.is_empty());
    }

    #[test]
    fn test_no_solution() {
        let nums = vec![1, 2, 3];
        let result = Solution::three_sum(nums);
        assert!(result.is_empty());
    }

    #[test]
    fn test_duplicates() {
        let nums = vec![-1, 0, 1, 0];
        let result = Solution::three_sum(nums);
        // Add your assertion
    }

    #[test]
    #[ignore] // Use this for slow/edge case tests
    fn test_large_input() {
        let nums = vec![0; 1000];
        let result = Solution::three_sum(nums);
        assert_eq!(result.len(), 1);
    }
}
```

**Run ignored tests:**
```bash
cargo test p0015 -- --ignored
```

### 7. Quick Testing in main.rs

For quick manual testing, edit `src/main.rs`:

```rust
use leetcode_solutions::problems::*;

fn main() {
    // Quick test
    let nums = vec![-1, 0, 1, 2, -1, -4];
    let result = p0015_three_sum::Solution::three_sum(nums);
    println!("Result: {:?}", result);
}
```

Run with: `cargo run`

### 8. Useful Patterns While Solving

**Use helper functions:**

```rust
impl Solution {
    pub fn three_sum(nums: Vec<i32>) -> Vec<Vec<i32>> {
        Self::helper(nums)
    }
    
    fn helper(mut nums: Vec<i32>) -> Vec<Vec<i32>> {
        // Your actual logic here
        vec![]
    }
}
```

**Common assertions for testing:**

```rust
// Check length
assert_eq!(result.len(), 2);

// Check if empty
assert!(result.is_empty());

// Check contains (for unordered results)
assert!(result.contains(&vec![-1, 0, 1]));

// Sort before comparing (if order doesn't matter)
let mut result = Solution::three_sum(nums);
result.sort();
let mut expected = vec![vec![-1, -1, 2], vec![-1, 0, 1]];
expected.sort();
assert_eq!(result, expected);
```

### 9. My Recommended Workflow

```bash
# Terminal 1: Watch mode
cargo watch -x "test p0015 -- --nocapture"

# Terminal 2: Quick runs when needed
cargo run

# Or use RustRover's built-in test runner:
# Right-click on test function → Run 'test_example1'
```

### 10. Common Issues & Solutions

**Issue: Tests pass but LeetCode fails**
- Add more edge cases
- Test with larger inputs
- Check integer overflow
- Test empty inputs

**Issue: Slow tests**
- Use `cargo test p0015 --release` for optimized build
- Profile with `#[ignore]` for slow tests

**Issue: Need to debug**
```rust
// Add this macro
dbg!(&nums);  // Prints with file:line info
```

---

**Try this workflow:**
1. Start `cargo watch -x "test p0015 -- --nocapture"`
2. Write one test case
3. Implement small piece
4. Save → see test result instantly
5. Iterate!

Does this help? Want me to show any specific part in more detail?