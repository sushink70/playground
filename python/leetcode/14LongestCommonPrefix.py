class TrieNode:
    def __init__(self) -> None:
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root

        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()

            node = node.children[char]

        node.is_end_of_word = True

    def search(self, word: str) -> bool:
        node = self.root

        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]

        return node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        node = self.root

        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]

        return True
    
    def get_all_words_with_prefix(self, prefix: str) -> list[str]:
        node = self.root

        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._dfs_collect(node, prefix, results)
        return results
    
    def _dfs_collect(self, node: TrieNode, current_word: str, results: list) -> None:
        if node.is_end_of_word:
            results.append(current_word)

        for char, child_node in node.children.items():
            self._dfs_collect(child_node, current_word + char, results) 


if __name__ == "__main__":
    trie = Trie()

    words = ["cat", "car", "card", "dog", "dodge", "door"]
    for word in words:
        trie.insert(word)

    print(f"Search 'car': {trie.search('car')}")