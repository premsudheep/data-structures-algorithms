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

public class DepthFirstSearch {

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    // 200. Number of Islands

    public int numIsLands(char[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;

        int count = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                count += dfsNumIslands(grid, i, j);
            }
        }
        return count;
    }

    private int dfsNumIslands(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0')
            return 0;

        grid[i][j] = '0';
        dfsNumIslands(grid, i - 1, j);
        dfsNumIslands(grid, i + 1, j);
        dfsNumIslands(grid, i, j - 1);
        dfsNumIslands(grid, i, j + 1);

        return 1;
    }

    // 1448. Count Good Nodes in Binary Tree
    // Given a binary tree root, a node X in the tree is named good if in the path
    // from root to X there are no nodes with a value greater than X.
    // Input: root = [3,1,4,3,null,1,5]
    // Output: 4
    // Explanation: Nodes in blue are good.
    // Root Node (3) is always a good node.
    // Node 4 -> (3,4) is the maximum value in the path starting from the root.
    // Node 5 -> (3,4,5) is the maximum value in the path
    // Node 3 -> (3,1,3) is the maximum value in the path.

    // Input: root = [3,3,null,4,2]
    // Output: 3
    // Explanation: Node 2 -> (3, 3, 2) is not good, because "3" is higher than it.

    private int numGoodNodes = 0;

    public int goodNodes(TreeNode root) {
        dfsGoodNodes(root, Integer.MIN_VALUE);
        return numGoodNodes;
    }

    private void dfsGoodNodes(TreeNode node, int maxSoFar) {
        if (maxSoFar <= node.val)
            numGoodNodes++;

        if (node.left != null)
            dfsGoodNodes(node.left, Math.max(node.val, maxSoFar));

        if (node.right != null)
            dfsGoodNodes(node.right, Math.max(node.val, maxSoFar));
    }

    // There are a total of numCourses courses you have to take, labeled from 0 to
    // numCourses - 1. You are given an array prerequisites where prerequisites[i] =
    // [ai, bi] indicates that you must take course bi first if you want to take
    // course ai.

    // For example, the pair [0, 1], indicates that to take course 0 you have to
    // first take course 1.
    // Return the ordering of courses you should take to finish all courses. If
    // there are many valid answers, return any of them. If it is impossible to
    // finish all courses, return an empty array.

    // Input: numCourses = 4, prerequisites = [[1,0],[2,0],[3,1],[3,2]]
    // Output: [0,2,1,3]
    // Explanation: There are a total of 4 courses to take. To take course 3 you
    // should have finished both courses 1 and 2. Both courses 1 and 2 should be
    // taken after you finished course 0.
    // So one correct course order is [0,1,2,3]. Another correct ordering is
    // [0,2,1,3].

    // Kahn's Algorith.

    public int[] findOrder(int numCourses, int[][] prerequisites) {

        Map<Integer, List<Integer>> adjList = new HashMap<>();
        int[] inDegree = new int[numCourses];
        int[] topologicalOrder = new int[numCourses];

        // Create the adjacency list representation of the graph
        for (int i = 0; i < prerequisites.length; i++) {
            int src = prerequisites[i][1];
            int dest = prerequisites[i][0];
            List<Integer> list = adjList.getOrDefault(src, new ArrayList<>());
            list.add(dest);
            adjList.putIfAbsent(src, list);

            // Record in-degree of each vertex
            inDegree[dest]++;
        }

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0)
                q.add(i);
        }

        int index = 0;
        // Process until the Q becomes empty
        // Breadth-First Search
        while (!q.isEmpty()) {
            int current = q.remove();
            topologicalOrder[index++] = current;

            // Reduce the in-degree of each neighbor by 1
            if (adjList.containsKey(current)) {
                for (int neigh : adjList.get(current)) {
                    inDegree[neigh]--;

                    // If in-degree of a neighbor becomes 0, add it to the Q
                    if (inDegree[neigh] == 0)
                        q.add(neigh);
                }
            }
        }

        if (index == numCourses)
            return topologicalOrder;

        return new int[0];
    }

    // 863. All Nodes Distance K in Binary Tree
    // Given the root of a binary tree, the value of a target node target, and an
    // integer k, return an array of the values of all nodes that have a distance k
    // from the target node.

    // https://www.youtube.com/watch?v=nPtARJ2cYrg

    Map<TreeNode, TreeNode> parent;

    public List<Integer> distanceK(TreeNode root, TreeNode target, int K) {
        parent = new HashMap<>();
        dfsDistanceK(root, null);

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(null);
        queue.add(target);

        Set<TreeNode> seen = new HashSet<>();
        seen.add(target);
        seen.add(null);

        int dist = 0;
        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            if (node == null) {
                if (dist == K) {
                    List<Integer> ans = new ArrayList<>();
                    for (TreeNode n : queue)
                        ans.add(n.val);
                    return ans;
                }
                queue.offer(null);
                dist++;
            } else {
                if (!seen.contains(node.left)) {
                    seen.add(node.left);
                    queue.offer(node.left);
                }
                if (!seen.contains(node.right)) {
                    seen.add(node.right);
                    queue.offer(node.right);
                }
                TreeNode par = parent.get(node);
                if (!seen.contains(par)) {
                    seen.add(par);
                    queue.offer(par);
                }
            }
        }

        return new ArrayList<Integer>();
    }

    public void dfsDistanceK(TreeNode node, TreeNode par) {
        if (node != null) {
            parent.put(node, par);
            dfsDistanceK(node.left, node);
            dfsDistanceK(node.right, node);
        }
    }

    // 124. Binary Tree Maximum Path Sum
    // A path in a binary tree is a sequence of nodes where each pair of adjacent
    // nodes in the sequence has an edge connecting them. A node can only appear in
    // the sequence at most once. Note that the path does not need to pass through
    // the root.

    // The path sum of a path is the sum of the node's values in the path.

    // Given the root of a binary tree, return the maximum path sum of any path.

    // Input: root = [-10,9,20,null,null,15,7]
    // Output: 42
    // Explanation: The optimal path is 15 -> 20 -> 7 with a path sum of 15 + 20 + 7
    // = 42.

    /*
     * 
     * 
     * 
     * 
     * 
     * 
     * 
     * 
     */

    // 236. Lowest Common Ancestor of a Binary Tree
    // Given a binary tree, find the lowest common ancestor (LCA) of two given nodes
    // in the tree.
    // The lowest common ancestor is defined between two nodes p and q as the lowest
    // node in T that has both p and q as descendants (where we allow a node to be a
    // descendant of itself).
    TreeNode lcaNode = null;

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        dfsLowestCommonAncestor(root, p, q);
        return lcaNode;
    }

    private boolean dfsLowestCommonAncestor(TreeNode currentNode, TreeNode p, TreeNode q) {
        // If reached the end of a branch, return false.
        if (currentNode == null)
            return false;

        // Left Recursion. If left recursion returns true, set left = 1 else 0
        int left = this.dfsLowestCommonAncestor(currentNode.left, p, q) ? 1 : 0;

        // Right Recursion
        int right = this.dfsLowestCommonAncestor(currentNode.right, p, q) ? 1 : 0;

        // If the current node is one of p or q
        int mid = (currentNode == p || currentNode == q) ? 1 : 0;

        // If any two of the flags left, right or mid become True
        if (mid + left + right >= 2) {
            this.lcaNode = currentNode;
        }

        // Return true if any one of the three bool values is True.
        return (mid + left + right > 0);
    }

    // 529. Minesweeper

    // You are given an m x n char matrix board representing the game board where:

    // 'M' represents an unrevealed mine,
    // 'E' represents an unrevealed empty square,
    // 'B' represents a revealed blank square that has no adjacent mines (i.e.,
    // above, below, left, right, and all 4 diagonals),
    // digit ('1' to '8') represents how many mines are adjacent to this revealed
    // square, and
    // 'X' represents a revealed mine.
    // You are also given an integer array click where click = [clickr, clickc]
    // represents the next click position among all the unrevealed squares ('M' or
    // 'E').

    // Return the board after revealing this position according to the following
    // rules:

    // If a mine 'M' is revealed, then the game is over. You should change it to
    // 'X'.
    // If an empty square 'E' with no adjacent mines is revealed, then change it to
    // a revealed blank 'B' and all of its adjacent unrevealed squares should be
    // revealed recursively.
    // If an empty square 'E' with at least one adjacent mine is revealed, then
    // change it to a digit ('1' to '8') representing the number of adjacent mines.
    // Return the board when no more squares will be revealed.

    public char[][] updateBoard(char[][] board, int[] click) {
        // once a mine is revealed, we can terminate immediately
        if (board[click[0]][click[1]] == 'M') {
            board[click[0]][click[1]] = 'X';
            return board;
        }

        reveal(board, click[0], click[1]);
        return board;
    }

    private void reveal(char[][] board, int i, int j) {
        // edge cases
        if (i < 0 || j < 0 || i >= board.length || j >= board[0].length || board[i][j] != 'E')
            return;

        board[i][j] = '0';
        int[][] neighbors = { { i - 1, j - 1 }, { i - 1, j }, { i - 1, j + 1 }, { i, j - 1 }, { i, j + 1 },
                { i + 1, j - 1 }, { i + 1, j }, { i + 1, j + 1 } };
        // check all neighbors for number of mines
        for (int[] neighbor : neighbors) {
            if (neighbor[0] < 0 || neighbor[1] < 0 || neighbor[0] >= board.length || neighbor[1] >= board[0].length)
                continue;
            if (board[neighbor[0]][neighbor[1]] == 'M')
                board[i][j]++;
        }

        if (board[i][j] != '0')
            return;

        // all neighbors are empty, recursively reveal them
        board[i][j] = 'B';
        for (int[] neighbor : neighbors)
            reveal(board, neighbor[0], neighbor[1]);
    }

    // 269 Alien Dictionary
    // Kahn's Algorithm

    // There is a new alien language that uses the English alphabet. However, the
    // order among the letters is unknown to you.

    // You are given a list of strings words from the alien language's dictionary,
    // where the strings in words are sorted lexicographically by the rules of this
    // new language.

    // Return a string of the unique letters in the new alien language sorted in
    // lexicographically increasing order by the new language's rules. If there is
    // no solution, return "". If there are multiple solutions, return any of them.

    // A string s is lexicographically smaller than a string t if at the first
    // letter where they differ, the letter in s comes before the letter in t in the
    // alien language. If the first min(s.length, t.length) letters are the same,
    // then s is smaller if and only if s.length < t.length.

    // Input: words = ["wrt","wrf","er","ett","rftt"]
    // Output: "wertf"

    // Input: words = ["z","x","z"]
    // Output: ""
    // Explanation: The order is invalid, so return "".

    public String alienOrder(String[] words) {

        // Step 0: Create data structures and find all unique letters.
        Map<Character, List<Character>> adjList = new HashMap<>();
        Map<Character, Integer> count = new HashMap<>();
        for (String s : words) {
            for (char c : s.toCharArray()) {
                count.put(c, 0);
                adjList.put(c, new ArrayList<>());
            }
        }

        // Step 1: Find the edges
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i];
            String word2 = words[i + 1];
            // Check that word2 is not a prefix of word1.
            if (word1.length() > word2.length() && word1.startsWith(word2)) {
                return "";
            }

            // Find the first non match and insert the corresponding relation.
            for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) {
                char ch1 = word1.charAt(j);
                char ch2 = word2.charAt(j);
                if (ch1 != ch2) {
                    adjList.get(ch1).add(ch2);
                    count.put(ch2, count.get(ch2) + 1);
                    break;
                }
            }
        }

        // Step 2: Breadth first search.
        StringBuilder sb = new StringBuilder();
        Queue<Character> queue = new LinkedList<>();
        for (char c : count.keySet()) {
            if (count.get(c).equals(0)) {
                queue.add(c);
            }
        }

        while (!queue.isEmpty()) {
            char ch = queue.remove();
            sb.append(ch);
            for (char next : adjList.get(ch)) {
                count.put(next, count.get(next) - 1);
                if (count.get(next).equals(0)) {
                    queue.add(next);
                }
            }
        }

        if (sb.length() < count.size()) {
            return "";
        }

        return sb.toString();
    }

    // 297. Serialize and Deserialize Binary Tree

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        return rSerialize(root, "");
    }

    private String rSerialize(TreeNode node, String str) {
        if (node == null)
            str += "null,";
        else {
            str += String.valueOf(node.val) + ",";
            str = rSerialize(node.left, str);
            str = rSerialize(node.right, str);
        }
        return str;
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        String[] dataArray = data.split(",");
        List<String> list = new LinkedList<>(Arrays.asList(dataArray));
        return rDeserialize(list);
    }

    private TreeNode rDeserialize(List<String> list) {
        if (list.get(0).equals("null")) {
            list.remove(0);
            return null;
        }

        TreeNode root = new TreeNode(Integer.valueOf(list.get(0)));
        list.remove(0);
        root.left = rDeserialize(list);
        root.right = rDeserialize(list);

        return root;
    }

    // 332. Reconstruct Itinerary

    // You are given a list of airline tickets where tickets[i] = [fromi, toi]
    // represent the departure and the arrival airports of one flight. Reconstruct
    // the itinerary in order and return it.

    // All of the tickets belong to a man who departs from "JFK", thus, the
    // itinerary must begin with "JFK". If there are multiple valid itineraries, you
    // should return the itinerary that has the smallest lexical order when read as
    // a single string.

    // For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than
    // ["JFK", "LGB"].
    // You may assume all tickets form at least one valid itinerary. You must use
    // all the tickets once and only once.

    // Input: tickets = [["MUC","LHR"],["JFK","MUC"],["SFO","SJC"],["LHR","SFO"]]
    // Output: ["JFK","MUC","LHR","SFO","SJC"]

    // Input: tickets =
    // [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
    // Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
    // Explanation: Another possible reconstruction is
    // ["JFK","SFO","ATL","JFK","ATL","SFO"] but it is larger in lexical order.

    // Refer to Eulerian Path Algorithm

    // 695. Max Area of Island
    // You are given an m x n binary matrix grid. An island is a group of 1's
    // (representing land) connected 4-directionally (horizontal or vertical.) You
    // may assume all four edges of the grid are surrounded by water.

    // The area of an island is the number of cells with a value 1 in the island.

    // Return the maximum area of an island in grid. If there is no island, return
    // 0.

    public int maxAreaOfIsland(int[][] grid) {
        int max = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                max = Math.max(max, dfsMaxAreaOfIsland(grid, i, j));
            }
        }
        return max;
    }

    private int dfsMaxAreaOfIsland(int[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == 0) {
            return 0;
        }
        grid[i][j] = 0;
        int count = 1;
        count += dfsMaxAreaOfIsland(grid, i + 1, j);
        count += dfsMaxAreaOfIsland(grid, i - 1, j);
        count += dfsMaxAreaOfIsland(grid, i, j + 1);
        count += dfsMaxAreaOfIsland(grid, i, j - 1);
        return count;
    }

    // 721. Accounts Merge

    // Given a list of accounts where each element accounts[i] is a list of strings,
    // where the first element accounts[i][0] is a name, and the rest of the
    // elements are emails representing emails of the account.

    // Now, we would like to merge these accounts. Two accounts definitely belong to
    // the same person if there is some common email to both accounts. Note that
    // even if two accounts have the same name, they may belong to different people
    // as people could have the same name. A person can have any number of accounts
    // initially, but all of their accounts definitely have the same name.

    // After merging the accounts, return the accounts in the following format: the
    // first element of each account is the name, and the rest of the elements are
    // emails in sorted order. The accounts themselves can be returned in any order.

    // Input: accounts =
    // [["Gabe","Gabe0@m.co","Gabe3@m.co","Gabe1@m.co"],["Kevin","Kevin3@m.co","Kevin5@m.co","Kevin0@m.co"],["Ethan","Ethan5@m.co","Ethan4@m.co","Ethan0@m.co"],["Hanzo","Hanzo3@m.co","Hanzo1@m.co","Hanzo0@m.co"],["Fern","Fern5@m.co","Fern1@m.co","Fern0@m.co"]]
    // Output:
    // [["Ethan","Ethan0@m.co","Ethan4@m.co","Ethan5@m.co"],["Gabe","Gabe0@m.co","Gabe1@m.co","Gabe3@m.co"],["Hanzo","Hanzo0@m.co","Hanzo1@m.co","Hanzo3@m.co"],["Kevin","Kevin0@m.co","Kevin3@m.co","Kevin5@m.co"],["Fern","Fern0@m.co","Fern1@m.co","Fern5@m.co"]]

    public List<List<String>> accountsMerge(List<List<String>> accounts) {

        Map<String, Set<String>> graph = new HashMap<>();
        Map<String, String> emailToName = new HashMap<>();

        buildAccountsGraph(graph, emailToName, accounts);

        List<List<String>> answer = new ArrayList<>();
        Set<String> visited = new HashSet<>();

        for (String email : emailToName.keySet()) {
            if (!visited.contains(email)) {
                visited.add(email);
                List<String> temp = new ArrayList<>();
                temp.add(email);
                dfsAccountsMerge(graph, temp, email, visited);

                Collections.sort(temp);
                temp.add(0, emailToName.get(email));
                answer.add(temp);
            }
        }
        return answer;
    }

    private void buildAccountsGraph(Map<String, Set<String>> graph, Map<String, String> emailToName,
            List<List<String>> accounts) {

        for (List<String> account : accounts) {
            String name = account.get(0);

            for (int i = 1; i < account.size(); i++) {
                String email = account.get(i);
                emailToName.put(email, name);
                graph.putIfAbsent(email, new HashSet<String>());

                if (i == 1)
                    continue;
                String prev = account.get(i - 1);
                graph.get(prev).add(email);
                graph.get(email).add(prev);
            }
        }
    }

    private void dfsAccountsMerge(Map<String, Set<String>> graph, List<String> temp, String email,
            Set<String> visited) {
        if (graph.get(email) == null || graph.get(email).size() == 0)
            return;

        for (String neighbour : graph.get(email)) {
            if (!visited.contains(neighbour)) {
                visited.add(neighbour);
                temp.add(neighbour);
                dfsAccountsMerge(graph, temp, neighbour, visited);
            }
        }
    }

    // 547. Number of Provinces

    // There are n cities. Some of them are connected, while some are not. If city a
    // is connected directly with city b, and city b is connected directly with city
    // c, then city a is connected indirectly with city c.

    // A province is a group of directly or indirectly connected cities and no other
    // cities outside of the group.

    // You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the
    // ith city and the jth city are directly connected, and isConnected[i][j] = 0
    // otherwise.

    // Return the total number of provinces.

    // Input: isConnected = [[1,1,0],[1,1,0],[0,0,1]]
    // Output: 2
    // Input: isConnected = [[1,0,0],[0,1,0],[0,0,1]]
    // Output: 3

    public int findCircleNum(int[][] isConnected) {
        int[] visited = new int[isConnected.length];
        int count = 0;
        Queue<Integer> queue = new LinkedList<>();

        for (int i = 0; i < isConnected.length; i++) {
            if (visited[i] == 0) {
                queue.add(i);
                while (!queue.isEmpty()) {
                    int s = queue.remove();
                    visited[s] = 1;
                    for (int j = 0; j < isConnected.length; j++) {
                        if (isConnected[s][j] == 1 && visited[j] == 0)
                            queue.add(j);
                    }
                }
                count++;
            }
        }
        return count;
    }

    // 987. Vertical Order Traversal of a Binary Tree

    // Given the root of a binary tree, calculate the vertical order traversal of
    // the binary tree.

    // For each node at position (row, col), its left and right children will be at
    // positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of
    // the tree is at (0, 0)

    Queue<int[]> nodeEntries;

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        nodeEntries = new PriorityQueue<>((e1, e2) -> {
            // 0 -> col, 1 -> row, 2 -> value
            for (int i = 0; i < e1.length; i++) {
                if (e1[i] != e2[i]) {
                    return e1[i] - e2[i];
                }
            }
            return 0;
        });

        dfsVerticalTraversal(root, 0, 0);

        List<List<Integer>> output = new ArrayList<>();
        int currentCol = Integer.MIN_VALUE;
        while (!nodeEntries.isEmpty()) {
            int[] entry = nodeEntries.remove();
            if (currentCol != entry[0]) {
                currentCol = entry[0];
                output.add(new ArrayList<>());
            }
            output.get(output.size() - 1).add(entry[2]);
        }
        return output;
    }

    private void dfsVerticalTraversal(TreeNode node, int col, int row) {
        if (node == null)
            return;

        nodeEntries.add(new int[] { col, row, node.val });
        dfsVerticalTraversal(node.left, col - 1, row + 1);
        dfsVerticalTraversal(node.right, col + 1, row + 1);
    }

    // https://www.youtube.com/watch?v=_426VVOB8Vo

    // You are given an n x n binary matrix grid. You are allowed to change at most
    // one 0 to be 1.

    // Return the size of the largest island in grid after applying this operation.

    // An island is a 4-directionally connected group of 1s.

    // Input: grid = [[1,1],[1,0]]
    // Output: 4
    // Explanation: Change the 0 to 1 and make the island bigger, only one island
    // with area = 4.

    int directions[][] = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };

    public int largestIsland(int[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;

        int max = 0;
        int islandId = 2;
        int m = grid.length;
        int n = grid[0].length;
        Map<Integer, Integer> map = new HashMap<>();

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                // extract the current size.
                // Also replace the current island 1's with islandId and increment and replace
                // other islands down the recursion.
                if (grid[i][j] == 1) {
                    int size = getIslandSize(grid, i, j, islandId);
                    map.put(islandId++, size);
                    max = Math.max(max, size);
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 0) {
                    // we don't want to sum up the same values from same islands
                    Set<Integer> set = new HashSet<>();
                    for (int[] direction : directions) {
                        int x = direction[0] + i;
                        int y = direction[1] + j;

                        if (x > -1 && y > -1 && x < m && y < n && grid[x][y] != 0) {
                            set.add(grid[x][y]); // adding islandId to the set
                        }
                    }

                    int sum = 1; // 0 -> 1
                    for (int num : set) {
                        int value = map.get(num);
                        sum += value;
                    }

                    max = Math.max(max, sum);
                }
            }
        }
        return max;
    }

    private int getIslandSize(int[][] grid, int i, int j, int islandId) {
        if (i < 0 || j < 0 || i >= grid.length || j >= grid[i].length || grid[i][j] != 1)
            return 0;

        grid[i][j] = islandId;
        int left = getIslandSize(grid, i, j - 1, islandId);
        int right = getIslandSize(grid, i, j + 1, islandId);
        int up = getIslandSize(grid, i - 1, j, islandId);
        int down = getIslandSize(grid, i + 1, j, islandId);

        return 1 + left + right + up + down; // +1 is the 0 that can be inserted.
    }

    // 199. Binary Tree Right Side View

    // Given the root of a binary tree, imagine yourself standing on the right side
    // of it, return the values of the nodes you can see ordered from top to bottom.

    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        traverse(root, list, 0);
        return list;
    }

    private void traverse(TreeNode node, List<Integer> list, int level) {
        if (node == null)
            return;
        if (level == list.size())
            list.add(node.val);
        if (node.right != null)
            traverse(node.right, list, level + 1);
        if (node.left != null)
            traverse(node.left, list, level + 1);
    }

    // 543. Diameter of Binary Tree

    // Given the root of a binary tree, return the length of the diameter of the
    // tree.

    // The diameter of a binary tree is the length of the longest path between any
    // two nodes in a tree. This path may or may not pass through the root.

    // The length of a path between two nodes is represented by the number of edges
    // between them.

    int diameter = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        longestPath(root);
        return diameter;
    }

    private int longestPath(TreeNode node) {
        if (node == null) {
            return 0;
        }

        int leftPath = longestPath(node.left);
        int rightPath = longestPath(node.right);

        diameter = Math.max(diameter, leftPath + rightPath);
        return 1 + Math.max(leftPath, rightPath);
    }

    // 472. Concatenated Words

    // Given an array of strings words (without duplicates), return all the
    // concatenated words in the given list of words.

    // A concatenated word is defined as a string that is comprised entirely of at
    // least two shorter words in the given array.

    // Input: words =
    // ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
    // Output: ["catsdogcats","dogcatsdog","ratcatdogcat"]
    // Explanation: "catsdogcats" can be concatenated by "cats", "dog" and "cats";
    // "dogcatsdog" can be concatenated by "dog", "cats" and "dog";
    // "ratcatdogcat" can be concatenated by "rat", "cat", "dog" and "cat".

    public List<String> findAllConcatenatedWordsInADict(String[] words) {
        List<String> result = new LinkedList<>();
        if (words.length == 0)
            return result;

        Set<String> wordSet = new HashSet<>();
        for (String word : words)
            wordSet.add(word);

        for (String word : words) {
            if (canCompose(word, wordSet)) {
                result.add(word);
            }
        }
        return result;
    }

    private boolean canCompose(String word, Set<String> wordSet) {
        if (word.length() == 0)
            return false;

        wordSet.remove(word);

        // dp[i]: for(word[1, .....i - 1]) if we can find some word(s) from wordSet to
        // be composed and generate this substr.
        boolean[] dp = new boolean[word.length() + 1];
        dp[0] = true;
        for (int i = 1; i <= word.length(); ++i) {
            for (int j = 0; j < i; ++j) {
                if (dp[j] && wordSet.contains(word.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        wordSet.add(word);
        return dp[word.length()];
    }

    // 329. Longest Increasing Path in a Matrix

    // Given an m x n integers matrix, return the length of the longest increasing
    // path in matrix.

    // From each cell, you can either move in four directions: left, right, up, or
    // down. You may not move diagonally or move outside the boundary (i.e.,
    // wrap-around is not allowed).

    // https://www.youtube.com/watch?v=uLjO2LUlLN4

    private static final int[][] dirs = { { 0, 1 }, { 1, 0 }, { 0, -1 }, { -1, 0 } };
    private int m, n;

    public int longestIncreasingPath(int[][] matrix) {
        if (matrix == null || matrix.length == 0)
            return 0;

        m = matrix.length;
        n = matrix[0].length;
        int[][] cache = new int[m][n];
        int ans = 0;

        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                ans = Math.max(ans, dfsLongestIncreasingPath(matrix, i, j, cache));
        return ans;
    }

    private int dfsLongestIncreasingPath(int[][] matrix, int i, int j, int[][] cache) {
        if (cache[i][j] > 0)
            return cache[i][j];

        int max = 0;
        for (int[] d : dirs) {
            int x = d[0] + i;
            int y = d[1] + j;
            if (x > -1 && x < m && y > -1 && y < n && matrix[x][y] > matrix[i][j]) {
                int longest = dfsLongestIncreasingPath(matrix, x, y, cache);
                max = Math.max(max, longest);
            }
        }

        cache[i][j] = max + 1;
        return cache[i][j];
    }

    // 426. Convert Binary Search Tree to Sorted Doubly Linked List

    TreeNode first = null;
    TreeNode last = null;

    public TreeNode treeToDoublyList(TreeNode root) {
        if (root == null)
            return null;

        treeToDoublyListRec(root);
        last.right = first;
        first.left = last;

        return first;

    }

    private void treeToDoublyListRec(TreeNode node) {
        if (node != null) {
            treeToDoublyListRec(node.left);
            if (last != null) {
                last.right = node; // bi-directional link for
                node.left = last; // for doubly linked list
            } else {
                first = node;
            }
            last = node;
            treeToDoublyListRec(node.right);
        }
    }

    // 98. Validate Binary Search Tree

    // Given the root of a binary tree, determine if it is a valid binary search
    // tree (BST).

    public boolean isValidBST(TreeNode root) {
        return dfsIsValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    private boolean dfsIsValidBST(TreeNode node, int min, int max) {
        if (node == null)
            return true;

        if (node.val < min || node.val > max)
            return false;

        return dfsIsValidBST(node.left, min, node.val - 1) && dfsIsValidBST(node.right, node.val + 1, max);
    }

}