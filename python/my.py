# /home/iamdreamer/Documents/github_sushink70/playground/dsa/python/my.py

def print_n_rec(s: str, n: int) -> None:
    """Print string s, n times, using recursion."""
    if n <= 0:
        return
    print(s)
    print_n_rec(s, n - 1)

def print_chars_rec(s: str, i: int = 0) -> None:
    """Print each character of s on its own line using recursion."""
    if i >= len(s):
        return
    print(s[i])
    print_chars_rec(s, i + 1)

if __name__ == "__main__":
    # Example: print "something" 5 times
    print_n_rec("something", 5)

    # Example: print each character of "something"
    print_chars_rec("something")