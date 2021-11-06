package com.prem.problems;

import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.HashMap;
import java.util.Queue;
import java.util.Set;
import java.util.LinkedList;
import java.util.HashSet;

public class BreadthFirstSearch {

    // This is the interface that allows for creating nested lists.
    // You should not implement it, or speculate about its implementation
    public class NestedInteger<T> {
        private T value;

        // Constructor initializes an empty nested list.
        public NestedInteger() {
        }

        // Constructor initializes a single integer.
        public NestedInteger(T value) {
            this.value = value;
        }

        // @return true if this NestedInteger holds a single integer, rather than a
        // nested list.
        public boolean isInteger() {
            return this.value instanceof Integer;
        }

        // @return the single integer that this NestedInteger holds, if it holds a
        // single integer
        // Return null if this NestedInteger holds a nested list
        public Integer getInteger() {
            return (int) this.value;
        }

        // Set this NestedInteger to hold a single integer.
        public void setInteger(T value) {
            this.value = value;
        }

        // Set this NestedInteger to hold a nested list and adds a nested integer to it.
        public void add(NestedInteger ni) {
            ((List<NestedInteger>) this.value).add(ni);
        }

        // @return the nested list that this NestedInteger holds, if it holds a nested
        // list
        // Return empty list if this NestedInteger holds a single integer
        public List<NestedInteger> getList() {
            return (List<NestedInteger>) this.value;
        }
    }

    public class Pair<X, Y> {
        private final X x;
        private final Y y;

        public Pair(X x, Y y) {
            this.x = x;
            this.y = y;
        }

        public X getKey() {
            return this.x;
        }

        public Y getValue() {
            return this.y;
        }

    }

    // 127. Word Ladder

    // A transformation sequence from word beginWord to word endWord using a
    // dictionary wordList is a sequence of words beginWord -> s1 -> s2 -> ... -> sk
    // such that:

    // Every adjacent pair of words differs by a single letter.
    // Every si for 1 <= i <= k is in wordList. Note that beginWord does not need to
    // be in wordList.
    // sk == endWord
    // Given two words, beginWord and endWord, and a dictionary wordList, return the
    // number of words in the shortest transformation sequence from beginWord to
    // endWord, or 0 if no such sequence exists.

    // Example 1:
    // Input: beginWord = "hit", endWord = "cog", wordList =
    // ["hot","dot","dog","lot","log","cog"]
    // Output: 5
    // Explanation: One shortest transformation sequence is "hit" -> "hot" -> "dot"
    // -> "dog" -> cog", which is 5 words long.

    // Example 2:
    // Input: beginWord = "hit", endWord = "cog", wordList =
    // ["hot","dot","dog","lot","log"]
    // Output: 0
    // Explanation: The endWord "cog" is not in wordList, therefore there is no
    // valid transformation sequence.

    public int ladderLength(String beginWord, String endWord, List<String> wordList) {

        // Since all words are of same length.
        int length = beginWord.length();

        // Dictionary to hold combinations of words that can be formed,
        // form any given word. By changing one letter at a time.
        Map<String, List<String>> allComboDict = new HashMap<>();

        wordList.forEach(word -> {
            for (int i = 0; i < length; i++) {
                // key is the generic word. (D*g)
                // value is a list of words which have the same intermediate generic word. (Dog,
                // Dig)
                String newWord = word.substring(0, i) + "*" + word.substring(i + 1, length);

                List<String> transformations = allComboDict.getOrDefault(newWord, new ArrayList<>());
                transformations.add(word);
                allComboDict.put(newWord, transformations);
            }
        });

        // Queue for BFS for shortest path.
        Queue<Pair<String, Integer>> q = new LinkedList<>();
        q.add(new Pair(beginWord, 1));

        // Visited to make sure we don't repeat processing same word.
        Map<String, Boolean> visited = new HashMap<>();
        visited.put(beginWord, true);

        while (!q.isEmpty()) {
            Pair<String, Integer> node = q.remove();
            String word = node.getKey();
            int level = node.getValue();

            for (int i = 0; i < length; i++) {
                // Intermediate words for current word
                String newWord = word.substring(0, i) + "*" + word.substring(i + 1, length);

                // Next states are all the words which share the same intermediate
                for (String adjWord : allComboDict.getOrDefault(newWord, new ArrayList<>())) {
                    // If at any point if we find what we are looking for
                    // i.e. the end word - we can return with the answer.
                    if (adjWord.equals(endWord)) {
                        return level + 1;
                    }
                    // Otherwise, add it to the BFS Queue. Also mark it visited
                    if (!visited.containsKey(adjWord)) {
                        visited.put(adjWord, true);
                        q.add(new Pair(adjWord, level + 1));
                    }
                }
            }
        }
        return 0;
    }

    // 909. Snakes and Ladders
    // Return the least number of moves required to reach the square n2. If it is
    // not possible to reach the square, return -1.

    public int snakesAndLadders(int[][] board) {
        int n = board.length;
        Set<String> visited = new HashSet<String>();
        Queue<int[]> pos = new LinkedList<int[]>();
        pos.offer(new int[] { n - 1, 0 });
        int count = 0;
        while (!pos.isEmpty()) {
            int size = pos.size();
            count++;
            while (size-- > 0) {
                int[] next = pos.poll();
                visited.add(next[0] + "," + next[1]);
                for (int i = 1; i <= 6; i++) {
                    int[] step = takeStep(board, next[0], next[1], i);
                    if (step[0] == n && step[1] == n)
                        return count;
                    if (board[step[0]][step[1]] != -1) {
                        step = getCord(n, board[step[0]][step[1]]);
                    }
                    if (step[0] == n && step[1] == n)
                        return count;
                    if (!visited.contains(step[0] + "," + step[1]))
                        pos.offer(step);
                }
            }
        }
        return -1;
    }

    /* Take steps at row, col */
    public int[] takeStep(int[][] board, int row, int col, int steps) {
        int n = board.length;
        int next = 0;
        if ((n + row) % 2 == 0)
            next = (n - row - 1) * n + n - col + steps;
        else
            next = (n - row - 1) * n + col + 1 + steps;
        return getCord(n, next);
    }

    private int[] getCord(int n, int value) {
        if (value >= n * n)
            return new int[] { n, n };
        int row = n - (value - 1) / n - 1;
        int col = 0;
        if ((n + row) % 2 == 0) {
            if (value % n == 0)
                col = 0;
            else
                col = n - ((value - 1) % n) - 1;
        } else {
            if (value % n == 0)
                col = n - 1;
            else
                col = ((value - 1) % n);
        }
        return new int[] { row, col };
    }

    // 1197. Minimum Knight Moves
    // In an infinite chess board with coordinates from -infinity to +infinity, you
    // have a knight at square [0, 0].

    // A knight has 8 possible moves it can make, as illustrated below. Each move is
    // two squares in a cardinal direction, then one square in an orthogonal
    // direction.

    // Return the minimum number of steps needed to move the knight to the square
    // [x, y]. It is guaranteed the answer exists.

    // Example 1:
    // Input: x = 2, y = 1
    // Output: 1
    // Explanation: [0, 0] → [2, 1]

    // Example 2:
    // Input: x = 5, y = 5
    // Output: 4
    // Explanation: [0, 0] → [2, 1] → [4, 2] → [3, 4] → [5, 5]

    private Map<String, Integer> memoKnightMoves = new HashMap<>();

    public int minKnightMoves(int x, int y) {
        return dfsKightMoves(Math.abs(x), Math.abs(y));
    }

    private int dfsKightMoves(int x, int y) {
        String key = x + "," + y;
        if (memoKnightMoves.containsKey(key))
            return memoKnightMoves.get(key);

        if (x + y == 0)
            return 0;
        else if (x + y == 2)
            return 2;
        int value = 1 + Math.min(dfsKightMoves(Math.abs(x - 1), Math.abs(y - 2)),
                dfsKightMoves(Math.abs(x - 2), Math.abs(y - 1)));
        memoKnightMoves.put(key, value);
        return value;

    }

    // 322. Coin Change

    // You are given an integer array coins representing coins of different
    // denominations and an integer amount representing a total amount of money.

    // Return the fewest number of coins that you need to make up that amount. If
    // that amount of money cannot be made up by any combination of the coins,
    // return -1.

    // You may assume that you have an infinite number of each kind of coin.

    /**
     * 
     * 
     * 
     * 
     * 
     */

    // 994. Rotting Oranges

    // You are given an m x n grid where each cell can have one of three values:

    // 0 representing an empty cell,
    // 1 representing a fresh orange, or
    // 2 representing a rotten orange.
    // Every minute, any fresh orange that is 4-directionally adjacent to a rotten
    // orange becomes rotten.

    // Return the minimum number of minutes that must elapse until no cell has a
    // fresh orange. If this is impossible, return -1.

    // Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
    // Output: 4

    public int orangesRotting(int[][] grid) {
        if (grid == null || grid.length == 0)
            return -1;

        int countFresh = 0;
        Queue<Pair<Integer, Integer>> queue = new LinkedList();
        int R = grid.length;
        int C = grid[0].length;

        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                if (grid[r][c] == 2)
                    queue.offer(new Pair(r, c));
                else if (grid[r][c] == 1)
                    countFresh++;

        if (countFresh == 0)
            return 0;

        int count = 0;
        int dirs[][] = { { 1, 0 }, { 0, 1 }, { -1, 0 }, { 0, -1 } };

        while (!queue.isEmpty()) {
            count++;
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                Pair<Integer, Integer> pair = queue.poll();
                int row = pair.getKey();
                int col = pair.getValue();
                for (int[] dir : dirs) {
                    int r = row + dir[0];
                    int c = col + dir[1];

                    if (r < 0 || r >= R || c < 0 || c >= C || grid[r][c] == 0 || grid[r][c] == 2)
                        continue;

                    grid[r][c] = 2;
                    queue.offer(new Pair(r, c));
                    countFresh--;
                }
            }
        }

        return countFresh == 0 ? count - 1 : -1;
    }

    // 364. Nested List Weight Sum II

    // You are given a nested list of integers nestedList. Each element is either an
    // integer or a list whose elements may also be integers or other lists.

    // The depth of an integer is the number of lists that it is inside of. For
    // example, the nested list [1,[2,2],[[3],2],1] has each integer's value set to
    // its depth. Let maxDepth be the maximum depth of any integer.

    // The weight of an integer is maxDepth - (the depth of the integer) + 1.

    // Return the sum of each integer in nestedList multiplied by its weight.

    // Input: nestedList = [[1,1],2,[1,1]]
    // Output: 8
    // Explanation: Four 1's with a weight of 1, one 2 with a weight of 2.
    // 1*1 + 1*1 + 2*2 + 1*1 + 1*1 = 8

    public int depthSumInverse(List<NestedInteger> nestedList) {
        if (nestedList == null)
            return 0;
        Queue<NestedInteger> queue = new LinkedList<>();
        int prev = 0;
        int total = 0;
        for (NestedInteger next : nestedList) {
            queue.offer(next);
        }

        while (!queue.isEmpty()) {
            int size = queue.size();
            int levelSum = 0;
            for (int i = 0; i < size; i++) {
                NestedInteger current = queue.poll();
                if (current.isInteger())
                    levelSum += current.getInteger();
                List<NestedInteger> nextList = current.getList();
                if (nextList != null) {
                    for (NestedInteger next : nextList) {
                        queue.offer(next);
                    }
                }
            }
            prev += levelSum;
            total += prev;
        }
        return total;
    }

}
