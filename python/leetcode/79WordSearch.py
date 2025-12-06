

from typing import List


class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        length_board = len(board[0]) * len(board)
        stack = []

        


        
        return True

board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]] 
word = "ABCCED"
s = Solution()
result = s.exist(board=board, word=word)
print(result)