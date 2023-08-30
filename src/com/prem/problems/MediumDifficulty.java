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
import java.util.Stack;
import java.util.LinkedList;
import java.util.HashSet;

// https://www.youtube.com/playlist?list=PLot-Xpze53lfOdF3KwpMSFEyfE77zIwiP

public class LeetcodeMediumDifficulty {

    // 15. 3Sum

    // Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
    // Notice that the solution set must not contain duplicate triplets.
    //  
    // Example 1:

    // Input: nums = [-1,0,1,2,-1,-4]
    // Output: [[-1,-1,2],[-1,0,1]]
    // Explanation: 
    // nums[0] + nums[1] + nums[2] = (-1) + 0 + 1 = 0.
    // nums[1] + nums[2] + nums[4] = 0 + 1 + (-1) = 0.
    // nums[0] + nums[3] + nums[4] = (-1) + 2 + (-1) = 0.
    // The distinct triplets are [-1,0,1] and [-1,-1,2].
    // Notice that the order of the output and the order of the triplets does not matter.

    // Example 2:

    // Input: nums = [0,1,1]
    // Output: []
    // Explanation: The only possible triplet does not sum up to 0.

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> list = new ArrayList<>();
        for(int i = 0; i < nums.length && nums[i] <=0 ; i++) {
            if(i == 0 || nums[i - 1] != nums[i]) {
                twoSum2(i, nums, list);
            }
        }
        return list;
    }

    private void twoSum2(int i, int[] nums, List<List<Integer>> list) {
        int lo = i + 1;
        int hi = nums.length - 1;
        while(lo < hi) {
            int sum = nums[i] + nums[lo] + nums[hi];
            if(sum < 0) {
                lo++;
            } else if (sum > 0) {
                hi--;
            } else {
                list.add(Arrays.asList(nums[i], nums[lo++], nums[hi--]));
                while(lo < hi && nums[lo] == nums[lo - 1]){
                    lo++;
                }
            }
        }
    }


    // 5. Longest Palindromic Substring

    //     Given a string s, return the longest palindromic substring in s.
    //  
    //     Example 1:

    //     Input: s = "babad"
    //     Output: "bab"
    //     Explanation: "aba" is also a valid answer.

    //     Example 2:

    //     Input: s = "cbbd"
    //     Output: "bb"

    public String longestPalindrome(String s) {
        String res = "";
        int resLen = 0;
        int l = 0;
        int r = 0;
        for(int i = 0; i < s.length(); i++) {
            l = i;
            r = i;
            // Odd length
            while(l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
                if((r - l + 1) > resLen){
                    res = s.substring(l, r + 1);
                    resLen = r - l + 1;
                }
                l -= 1;
                r += 1;
            }

            // Even length
            l = i;
            r = i + 1;
            while(l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r))  {
                if((r - l + 1) > resLen){
                    res = s.substring(l, r + 1);
                    resLen = r - l + 1;
                }
                l -= 1;
                r += 1;
            }
        }
        return res;
    }


    // 46. Permutations

    // Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

    // Example 1:

    // Input: nums = [1,2,3]
    // Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
    // Example 2:

    // Input: nums = [0,1]
    // Output: [[0,1],[1,0]]

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        backtrackPerm(result, new ArrayList<>(), nums);
        return result;
    }

    private void backtrackPerm(List<List<Integer>> result, List<Integer> tempList, int[] nums) {
        if(tempList.size() == nums.length) {
            result.add(new ArrayList<>(tempList));
        } else {
            for(int i = 0; i < nums.length; i++) {
                if(tempList.contains(nums[i])) {
                    continue;
                }
                tempList.add(nums[i]);
                backtrackPerm(result, tempList, nums);
                tempList.remove(tempList.size() - 1);
            }
        }
    }

    // 3. Longest Substring Without Repeating Characters

    // Given a string s, find the length of the longest substring without repeating characters.

    // Example 1:

    // Input: s = "abcabcbb"
    // Output: 3
    // Explanation: The answer is "abc", with the length of 3.
    // Example 2:

    // Input: s = "bbbbb"
    // Output: 1
    // Explanation: The answer is "b", with the length of 1.
    // Example 3:

    // Input: s = "pwwkew"
    // Output: 3
    // Explanation: The answer is "wke", with the length of 3.
    // Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.

    public int lengthOfLongestSubstring(String s) {
        int a = 0;
        int b = 0;
        int max = 0;
        Set<Character> set = new HashSet<>();
        while (b < s.length()) {
            if (set.contains(s.charAt(b))) {
                set.remove(s.charAt(a));
                a++;
            } else {
                set.add(s.charAt(b));
                b++;
                max = Math.max(max, set.size());
            }
        }
        return max;
    }

    // 152. Maximum Product Subarray

    // Given an integer array nums, find a subarray that has the largest product, and return the product.

    // The test cases are generated so that the answer will fit in a 32-bit integer.

    // Example 1:

    // Input: nums = [2,3,-2,4]
    // Output: 6
    // Explanation: [2,3] has the largest product 6.
    // Example 2:

    // Input: nums = [-2,0,-1]
    // Output: 0
    // Explanation: The result cannot be 2, because [-2,-1] is not a subarray.

    public int maxProduct(int[] nums) {
        int maxSoFar = nums[0];
        int minSoFar = nums[0];
        int result = maxSoFar;
        for(int i = 1; i < nums.length; i++) {
            int current = nums[i];
            int temp = Math.max(current, Math.max(current * maxSoFar, current * minSoFar));
            minSoFar = Math.min(current, Math.min(current * maxSoFar, current * minSoFar));
            maxSoFar = temp;
            result = Math.max(result, maxSoFar);
        }
        return result;
    }

    // 322. Coin Change

    // You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

    // Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

    // You may assume that you have an infinite number of each kind of coin.    

    // Example 1:

    // Input: coins = [1,2,5], amount = 11
    // Output: 3
    // Explanation: 11 = 5 + 5 + 1

    // private static final int INF = Integer.MAX_VALUE;
    // public int coinChange(int[] coins, int amount) {
    //     if(coins == null) {
    //         return -1;
    //     }
    //     if(amount == 0) {
    //         return 0;
    //     }
    //     int rows = coins.length - 1;
    //     int columns = amount;
    //     int[][] dp = new int[rows + 1][columns + 1];
    //     java.util.Arrays.fill(dp[0], INF);
    //     dp[0][0] = 0;

    //     for(int i = 1; i <= rows; i++) {
    //         int coinValue = coins[i - 1];
    //         for(int j = 1; j <= columns; j++) {
    //             dp[i][j] = dp[i - 1][j];
    //             if(j - coinValue >= 0 && dp[i][j - coinValue] + 1 < dp[i][j]) {
    //                 dp[i][j] = dp[i][j - coinValue] + 1;
    //             }
    //         }
    //     }

    //     if(dp[rows][columns] == INF) {
    //         return -1;
    //     }
    //     return dp[rows][columns];
    // }

    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount+1];
        Arrays.fill(dp,amount + 1);
        dp[0] = 0;
        for(int i = 1; i < amount + 1; i++)
        {
            for(int coin : coins)
            {
                if(i - coin >= 0)
                {
                    dp[i] = Math.min(dp[i], dp[i-coin] + 1);
                }
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    // 2. Add Two Numbers

    // You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

    // You may assume the two numbers do not contain any leading zero, except the number 0 itself.

    // Example 1:

    // Input: l1 = [2,4,3], l2 = [5,6,4]
    // Output: [7,0,8]
    // Explanation: 342 + 465 = 807.

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1);
        ListNode current = dummy;
        int carry = 0;
        while(l1 != null || l2 != null) {
            int val1 = (l1 != null ) ? l1.val : 0;
            int val2 = (l2 != null ) ? l2.val : 0;

            int sum = val1 + val2 + carry;
            carry = sum / 10;
            sum = sum % 10;
            current.next = new ListNode(sum);
            current = current.next;

            if(l1 != null) {
                l1 = l1.next;
            }
            if(l2 != null) {
                l2 = l2.next;
            }
        }

        if(carry != 0) {
            current.next = new ListNode(carry);
            current = current.next;
        }

        return dummy.next;
    }

    // 300. Longest Increasing Subsequence

    // Given an integer array nums, return the length of the longest strictly increasing subsequence.

    // Example 1:

    // Input: nums = [10,9,2,5,3,7,101,18]
    // Output: 4
    // Explanation: The longest increasing subsequence is [2,3,7,101], therefore the length is 4.

    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        java.util.Arrays.fill(dp, 1);

        for(int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if(nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int longest = 0;
        for(int c: dp) {
            longest = Math.max(c, longest);
        }
        return longest;
    }


    // 207. Course Schedule

    // There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    // For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    // Return true if you can finish all courses. Otherwise, return false.

    // Example 1:

    // Input: numCourses = 2, prerequisites = [[1,0]]
    // Output: true
    // Explanation: There are a total of 2 courses to take. 
    // To take course 1 you should have finished course 0. So it is possible.

    public boolean canFinish(int numCourses, int[][] prerequisites) {
        Map<Integer, List<Integer>> adjList = new HashMap<>();
        int[] inDegree = new int[numCourses];
        int[] topologicalOrder = new int[numCourses];

        for(int i = 0; i < prerequisites.length; i++) {
            int[] course = prerequisites[i];
            int src = course[1];
            int dest = course[0];
            List list = adjList.getOrDefault(src, new ArrayList<>());
            list.add(dest);
            adjList.putIfAbsent(src, list);
            inDegree[dest]++;
        }

        Queue<Integer> q = new LinkedList<>();
        for(int i = 0; i < inDegree.length; i++) {
            if(inDegree[i] == 0) {
                q.add(i);
            }
        }

        int index = 0;
        while(!q.isEmpty()) {
            int current = q.remove();
            topologicalOrder[index++] = current;
            if(adjList.containsKey(current)) {
                List<Integer> neighbours = adjList.get(current);
                for(int neigh: neighbours) {
                    inDegree[neigh]--;
                    if(inDegree[neigh] == 0) {
                        q.add(neigh);
                    }
                }
            }
        }

        if(index != numCourses) {
            return false;
        }
        return true;
    }

    // 139. Word Break

    // Given a string s and a dictionary of strings wordDict, return true if s can be segmented into a space-separated sequence of one or more dictionary words.

    // Note that the same word in the dictionary may be reused multiple times in the segmentation.

    // Example 1:

    // Input: s = "leetcode", wordDict = ["leet","code"]
    // Output: true
    // Explanation: Return true because "leetcode" can be segmented as "leet code".
    // Example 2:

    // Input: s = "applepenapple", wordDict = ["apple","pen"]
    // Output: true
    // Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
    // Note that you are allowed to reuse a dictionary word.
    // Example 3:

    // Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
    // Output: false

    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> dict = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;

        for(int i = 1; i <= n; i++) {
            for(int j = 0; j < i; j++) {
                if(dp[j] && dict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        return dp[n];
    }

    // 33. Search in Rotated Sorted Array

    // There is an integer array nums sorted in ascending order (with distinct values).

    // Prior to being passed to your function, nums is possibly rotated at an unknown pivot index k (1 <= k < nums.length) such that the resulting array is [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]] (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3 and become [4,5,6,7,0,1,2].

    // Given the array nums after the possible rotation and an integer target, return the index of target if it is in nums, or -1 if it is not in nums.

    // You must write an algorithm with O(log n) runtime complexity.

    

    // Example 1:

    // Input: nums = [4,5,6,7,0,1,2], target = 0
    // Output: 4
    // Example 2:

    // Input: nums = [4,5,6,7,0,1,2], target = 3
    // Output: -1
    // Example 3:

    // Input: nums = [1], target = 0
    // Output: -1

    public int search(int[] nums, int target) {
        int start = 0, end = nums.length - 1;
        while (start <= end) {
            int mid = start + (end - start) / 2;
            if (nums[mid] == target)
                return mid;
            else if (nums[mid] >= nums[start]) {
                if (target >= nums[start] && target < nums[mid])
                    end = mid - 1;
                else
                    start = mid + 1;
            } else {
                if (target <= nums[end] && target > nums[mid])
                    start = mid + 1;
                else
                    end = mid - 1;
            }
        }
        return -1;
    }

    // 22. Generate Parentheses

    // Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

    // Example 1:

    // Input: n = 3
    // Output: ["((()))","(()())","(())()","()(())","()()()"]
    // Example 2:

    // Input: n = 1
    // Output: ["()"]

    List<String> result = new ArrayList<>();
    public List<String> generateParenthesis(int n) {
        // only add open paranthesis if open < n
        // only add a closing paranthesis if closed < open
        // valid if open == closed == n
        backtackParanthesis(0, 0, n, "");
        return result;
    }

    private void backtackParanthesis(int left, int right, int n, String s) {
        if(s.length() == n * 2) {
            result.add(s);
            return;
        }

        if(left < n) {
            backtackParanthesis(left + 1, right, n, s + "(");
        }

        if(right < left) {
            backtackParanthesis(left, right + 1, n, s + ")");
        }
    }

    // 200. Number of Islands

    // Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

    // An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

    // Example 1:

    // Input: grid = [
    // ["1","1","1","1","0"],
    // ["1","1","0","1","0"],
    // ["1","1","0","0","0"],
    // ["0","0","0","0","0"]
    // ]
    // Output: 1

    public int numIslands(char[][] grid) {
        if(grid == null || grid.length == 0) {
            return 0;
        }
        int count = 0;
        for(int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                count += islandDFS(i, j, grid);
            }
        }
        return count;
    }

    private int islandDFS(int i, int j, char[][] grid) {
        if(i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0') {
            return 0;
        }
        grid[i][j] = '0';
        islandDFS(i - 1, j, grid);
        islandDFS(i + 1, j, grid);
        islandDFS(i, j - 1, grid);
        islandDFS(i, j + 1, grid);
        return 1;
    }

    // 11. Container With Most Water

    // You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

    // Find two lines that together with the x-axis form a container, such that the container contains the most water.

    // Return the maximum amount of water a container can store.

    // Notice that you may not slant the container.

    // Example 1:

    // Input: height = [1,8,6,2,5,4,8,3,7]
    // Output: 49
    // Explanation: The above vertical lines are represented by array [1,8,6,2,5,4,8,3,7]. In this case, the max area of water (blue section) the container can contain is 49.

    public int maxArea(int[] height) {
        int L = 0;
        int R = height.length - 1;
        int max = 0;
        while(L < R) {
            int min = Math.min(height[L], height[R]);
            max = Math.max(max, min * (R - L));
            if(height[L] < height[R]) {
                L++;
            } else {
                R--;
            }
        }
        return max;
    }

    // 48. Rotate Image

    // You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).

    // You have to rotate the image in-place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.    

    // Example 1:

    // Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    // Output: [[7,4,1],[8,5,2],[9,6,3]]

    public void rotate(int[][] matrix) {
        int l = 0;
        int r = matrix.length - 1;

        while(l < r) {
            for (int i = l; i < r; i++) {
                int top = l;
                int bottom = r;
                int temp = matrix[top][l + i];
                matrix[top][l + i] = matrix[bottom - i][l];
                matrix[bottom - i][l] = matrix[bottom][r - i];
                matrix[bottom][r - i] = matrix[top + i][r];
                matrix[top + i][r] = temp;
            }
            l++;
            r--;
        }
    }

    // 79. Word Search

    // Given an m x n grid of characters board and a string word, return true if word exists in the grid.

    // The word can be constructed from letters of sequentially adjacent cells, where adjacent cells are horizontally or vertically neighboring. The same letter cell may not be used more than once.
    
    // Example 1:

    // Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word = "ABCCED"
    // Output: true

    private char[][] board;
    private int ROWS;
    private int COLUMNS;
    public boolean exist(char[][] board, String word) {
        this.board = board;
        this.ROWS = board.length;
        this.COLUMNS = board[0].length;
        for(int row = 0; row < this.ROWS; row++) {
           for(int col = 0; col < this.COLUMNS; col++) {
               if(backTrackingWord(row, col, word, 0)) {
                   return true;
               }
           }
        }
        return false;
    }

    private boolean backTrackingWord(int row, int col, String word, int wordIndex) {
        if(wordIndex >= word.length()) {
            return true;
        }

        if(row < 0 || row >= this.board.length || col < 0 || col >= this.board[row].length || this.board[row][col] != word.charAt(wordIndex)) {
            return false;
        }

        boolean result = false;
        this.board[row][col] = '#';
        int[] rowOffsets = {0, 1, 0, -1};
        int[] colOffsets = {1, 0, -1, 0};
        for(int d = 0; d < 4; d++) {
            result = backTrackingWord(row + rowOffsets[d], col + colOffsets[d], word, wordIndex + 1);
            if(result) {
                break;
            }
        }
        this.board[row][col] = word.charAt(wordIndex);
        return result;
    }

    // 78. Subsets

    // Given an integer array nums of unique elements, return all possible subsets (the power set).

    // The solution set must not contain duplicate subsets. Return the solution in any order.

    // Example 1:

    // Input: nums = [1,2,3]
    // Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> ans = new ArrayList<>();
        backtrackSubsets(0, nums, new ArrayList<Integer>(), ans);
        return ans;
    }

    private void backtrackSubsets(int index, int[] nums, ArrayList<Integer> temp, List<List<Integer>> ans) {
        ans.add(new ArrayList<Integer>(temp));
        for(int i = index; i < nums.length; i++) {
            if(i != index && nums[i] == nums[i - 1]) {
                continue;
            }
            temp.add(nums[i]);
            backtrackSubsets(i + 1, nums, temp, ans);
            temp.remove(temp.size() - 1);
        }
    }

    // 39. Combination Sum

    // Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

    // The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

    // The test cases are generated such that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

    // Example 1:

    // Input: candidates = [2,3,6,7], target = 7
    // Output: [[2,2,3],[7]]
    // Explanation:
    // 2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
    // 7 is a candidate, and 7 = 7.
    // These are the only two combinations.

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> l = new ArrayList<>();
        List<Integer> l1 = new ArrayList<>();
        combinationSumDFS(0, candidates, target, l, l1);
        return l;
    }

    public void combinationSumDFS(int ind, int[] arr, int target, List<List<Integer>> l, List<Integer> l1) {
        if(target == 0) {
            l.add(new ArrayList<>(l1));
            return;
        }
        if(ind == arr.length){
            return;
        }
        if(arr[ind] <= target) {
            l1.add(arr[ind]);
            combinationSumDFS(ind, arr, target - arr[ind], l, l1);
            l1.remove(l1.size() - 1);
        }
        combinationSumDFS(ind + 1, arr, target, l, l1);
    }

    // 47. Permutations II

    // Given a collection of numbers, nums, that might contain duplicates, return all possible unique permutations in any order. 

    // Example 1:

    // Input: nums = [1,1,2]
    // Output:
    // [[1,1,2],
    // [1,2,1],
    // [2,1,1]]
    // Example 2:

    // Input: nums = [1,2,3]
    // Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> list = new ArrayList<>();
        List<Integer> currlist = new ArrayList<>();
        Map<Integer,Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        permuteUniqueHelper(nums,list,currlist,map);
        return list;
    }

    private void permuteUniqueHelper(int[] nums, List<List<Integer>> list, List<Integer> currlist, Map<Integer, Integer> map) {
        if(currlist.size()==nums.length){
            if(!list.contains(currlist)) {
                list.add(new ArrayList<>(currlist));
            }
            return;
        }
        for(int i = 0; i < nums.length; i++){
            if(map.get(nums[i]) > 0 ){
                currlist.add(nums[i]);
                map.put(nums[i], map.get(nums[i]) - 1);
                permuteUniqueHelper(nums, list, currlist, map);
                map.put(nums[i], map.get(nums[i]) + 1);
                currlist.remove(currlist.size() - 1);
            }
        }
    }

    // 1143. Longest Common Subsequence

    // Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

    // A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

    // For example, "ace" is a subsequence of "abcde".
    // A common subsequence of two strings is a subsequence that is common to both strings.

    // Example 1:

    // Input: text1 = "abcde", text2 = "ace" 
    // Output: 3  
    // Explanation: The longest common subsequence is "ace" and its length is 3.

    public int longestCommonSubsequence(String text1, String text2) {
        int rows = text1.length();
        int columns = text2.length();
        int[][] dp = new int[rows + 1][columns + 1];
        
        java.util.Arrays.fill(dp[0], 0);
        for (int i = 0; i < dp.length; i++) {
            Arrays.fill(dp[i], 0);
        }

        for(int i = 1; i <= rows; i++) {
            for(int j = 1; j <= columns; j++) {
                if(text1.charAt(i - 1) == text2.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[rows][columns];
    }

    // 73. Set Matrix Zeroes
    // Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
    // You must do it in place.

    // Example 1:

    // Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
    // Output: [[1,0,1],[0,0,0],[1,0,1]]

    public void setZeroes(int[][] matrix) {
        int rlength = matrix.length;
        int clength = matrix[0].length;
        Set<Integer> rows = new HashSet<>();
        Set<Integer> cols = new HashSet<>();

        // Essentially, we mark the rows and columns that are to be made zero
        for (int i = 0; i < rlength; i++) {
            for (int j = 0; j < clength; j++) {
                if (matrix[i][j] == 0) {
                    rows.add(i);
                    cols.add(j);
                }
            }
        }

        // Iterate over the array once again and using the rows and cols sets, update
        // the elements.
        for (int i = 0; i < rlength; i++) {
            for (int j = 0; j < clength; j++) {
                if (rows.contains(i) || cols.contains(j)) {
                    matrix[i][j] = 0;
                }
            }
        }
    }

    // 210. Course Schedule II

    // There are a total of numCourses courses you have to take, labeled from 0 to numCourses - 1. You are given an array prerequisites where prerequisites[i] = [ai, bi] indicates that you must take course bi first if you want to take course ai.

    // For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
    // Return the ordering of courses you should take to finish all courses. If there are many valid answers, return any of them. If it is impossible to finish all courses, return an empty array.

    // Example 1:

    // Input: numCourses = 2, prerequisites = [[1,0]]
    // Output: [0,1]
    // Explanation: There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1].

    public int[] findOrder(int numCourses, int[][] prerequisites) {
        int n = prerequisites.length;

        ArrayList<ArrayList<Integer>> adjList = new ArrayList<>();
        for(int i=0;i<numCourses;i++){
            adjList.add(new ArrayList<>());
        }

        for(int i=0;i<n;i++){
            adjList.get(prerequisites[i][0]).add(prerequisites[i][1]);
        }

        int[] in = new int[numCourses];
        for(int i=0;i<numCourses;i++){
            for(int it: adjList.get(i)){
                in[it]++;
            }
        }

        Queue<Integer> q = new LinkedList<>();
        for(int i=0;i<numCourses;i++){
            if(in[i]==0) q.add(i);
        }

        ArrayList<Integer> res = new ArrayList<Integer>();
        while(!q.isEmpty()){
            int node = q.poll();

            res.add(node);
            for(int i: adjList.get(node)){
                in[i]--;
                if(in[i]==0) q.add(i);
            }
        }
        int[] ans = new int[res.size()];
        for(int i=res.size()-1;i>=0;i--){
            ans[ans.length-i-1]=res.get(i);
        }
        if(res.size()!=numCourses) return new int[0];
        return ans;
    }

    // 105. Construct Binary Tree from Preorder and Inorder Traversal

    // Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

    // Example 1:

    // Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    // Output: [3,9,20,null,null,15,7]

    // Example 2:

    // Input: preorder = [-1], inorder = [-1]
    // Output: [-1]

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode() {}
        TreeNode(int val) { this.val = val; }
        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    int preorderIndex;
    Map<Integer, Integer> inOrderIndexMap;
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        preorderIndex = 0;
        inOrderIndexMap = new HashMap<>();
        for(int i = 0; i < inorder.length; i++) {
            inOrderIndexMap.put(inorder[i], i);
        }
        return arrayToTree(preorder, 0, preorder.length - 1);
    }

    private TreeNode arrayToTree(int[] preorder, int left, int right) {
        if(left > right) {
            return null;
        }
        int rootValue = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootValue);

        root.left = arrayToTree(preorder, left, inOrderIndexMap.get(rootValue) - 1);
        root.right = arrayToTree(preorder, inOrderIndexMap.get(rootValue) + 1, right);
        return root;
    }

    // 55. Jump Game

    // You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

    // Return true if you can reach the last index, or false otherwise.

    // Example 1:

    // Input: nums = [2,3,1,1,4]
    // Output: true
    // Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
    // Example 2:

    // Input: nums = [3,2,1,0,4]
    // Output: false
    // Explanation: You will always arrive at index 3 no matter what. Its maximum jump length is 0, which makes it impossible to reach the last index.

    public boolean canJump(int[] nums) {
        int goal = nums.length - 1;
        for(int i = goal; i >= 0; i--) {
            if(i + nums[i] >= goal){
                goal = i;
            }
        }
        return goal == 0;
    }

    // 98. Validate Binary Search Tree

    // Given the root of a binary tree, determine if it is a valid binary search tree (BST).

    // A valid BST is defined as follows:

    // The left subtree of a node contains only nodes with keys less than the node's key.
    // The right subtree of a node contains only nodes with keys greater than the node's key.
    // Both the left and right subtrees must also be binary search trees.

    // Example 1:

    // Input: root = [2,1,3]
    // Output: true

    public boolean isValidBST(TreeNode root) {
        return isValidBSTDFS(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
    }

    private boolean isValidBSTDFS(TreeNode node, int min, int max) {
        if(node == null) {
            return true;
        }
        if(node.val < min || node.val > max) {
            return false;
        }
        return isValidBSTDFS(node.left, min, node.val - 1) && isValidBSTDFS(node.right, node.val + 1, max);
    }



    // 146. LRU Cache

    // Design a data structure that follows the constraints of a Least Recently Used (LRU) cache.

    // Implement the LRUCache class:

    // LRUCache(int capacity) Initialize the LRU cache with positive size capacity.
    // int get(int key) Return the value of the key if the key exists, otherwise return -1.
    // void put(int key, int value) Update the value of the key if the key exists. Otherwise, add the key-value pair to the cache. If the number of keys exceeds the capacity from this operation, evict the least recently used key.
    // The functions get and put must each run in O(1) average time complexity.

    // Example 1:

    // Input
    // ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
    // [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
    // Output
    // [null, null, null, 1, null, -1, null, -1, 3, 4]

    // Explanation
    // LRUCache lRUCache = new LRUCache(2);
    // lRUCache.put(1, 1); // cache is {1=1}
    // lRUCache.put(2, 2); // cache is {1=1, 2=2}
    // lRUCache.get(1);    // return 1
    // lRUCache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
    // lRUCache.get(2);    // returns -1 (not found)
    // lRUCache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
    // lRUCache.get(1);    // return -1 (not found)
    // lRUCache.get(3);    // return 3
    // lRUCache.get(4);    // return 4

    // Explaination: https://www.youtube.com/watch?v=7ABFKPK2hD4&list=PLot-Xpze53lfOdF3KwpMSFEyfE77zIwiP&index=27

    /**
     * Your LRUCache object will be instantiated and called as such:
     * LRUCache obj = new LRUCache(capacity);
     * int param_1 = obj.get(key);
     * obj.put(key,value);
     */

    class LRUCache {
        class Node {
            int key;
            int val;
            Node prev;
            Node next;

            Node(int key, int val) {
                this.key = key;
                this.val = val;
            }
        }

        Node head = new Node(-1, -1);
        Node tail = new Node(-1, -1);
        int cap;
        HashMap<Integer, Node> m = new HashMap<>();

        public LRUCache(int capacity) {
            cap = capacity;
            head.next = tail;
            tail.prev = head;
        }

        private void addNode(Node newnode) {
            Node temp = head.next;

            newnode.next = temp;
            newnode.prev = head;

            head.next = newnode;
            temp.prev = newnode;
        }

        private void deleteNode(Node delnode) {
            Node prevv = delnode.prev;
            Node nextt = delnode.next;

            prevv.next = nextt;
            nextt.prev = prevv;
        }

        public int get(int key) {
            if (m.containsKey(key)) {
                Node resNode = m.get(key);
                int ans = resNode.val;

                m.remove(key);
                deleteNode(resNode);
                addNode(resNode);

                m.put(key, head.next);
                return ans;
            }
            return -1;
        }

        public void put(int key, int value) {
            if (m.containsKey(key)) {
                Node curr = m.get(key);
                m.remove(key);
                deleteNode(curr);
            }

            if (m.size() == cap) {
                m.remove(tail.prev.key);
                deleteNode(tail.prev);
            }

            addNode(new Node(key, value));
            m.put(key, head.next);
        }
    }

    // 131. Palindrome Partitioning

    // Given a string s, partition s such that every substring of the partition is a palindrome. Return all possible palindrome partitioning of s.

    // Example 1:

    // Input: s = "aab"
    // Output: [["a","a","b"],["aa","b"]]
    // Example 2:

    // Input: s = "a"
    // Output: [["a"]]

    int n;
    boolean[][] is_palindrome;
    String[][] substrings;
    List<List<String>> ans;

    public List<List<String>> partition(String s) {
        n = s.length();
        is_palindrome = new boolean[n + 1][n + 1];
        substrings = new String[n + 1][n + 1];
        for (int i = 0; i < n; i++)  {
            for (int j = i + 1; j <= n; j++) {
                substrings[i][j] = s.substring(i, j);
                is_palindrome[i][j] = isPalindromeHelper(substrings[i][j]);
            }
        }

        ans = new ArrayList<List<String>>();
        findSubstrings(0, new ArrayList<String>());
        return ans;
    }

    void findSubstrings(int ind, ArrayList<String> list) {
        if (ind == n) {
            ans.add(new ArrayList<String>(list));
            return;
        }

        for (int i = ind + 1; i <= n; i++) {
            if (!is_palindrome[ind][i]) continue;
            list.add(substrings[ind][i]);
            findSubstrings(i, list);
            list.remove(list.size() - 1);
        }
    }

    boolean isPalindromeHelper(String s) {
        int lower = 0;
        int higher = s.length() - 1;
        while (lower < higher) {
            if (s.charAt(lower) != s.charAt(higher)) return false;
            lower++;
            higher--;
        }
        return true;
    }

    // 424. Longest Repeating Character Replacement

    // You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.

    // Return the length of the longest substring containing the same letter you can get after performing the above operations.

    // Example 1:

    // Input: s = "ABAB", k = 2
    // Output: 4
    // Explanation: Replace the two 'A's with two 'B's or vice versa.

    public int characterReplacement(String s, int k) {
        int start = 0;
        int end = 0;
        int maxCount = 0;
        int maxSize = 0;
        int[] chars = new int[26];
        for(; end < s.length(); end++) {
            chars[s.charAt(end) - 'A']++;
            maxCount = Math.max(maxCount, chars[s.charAt(end) - 'A']);

            if((end - start + 1) - maxCount > k) {
                chars[s.charAt(start) - 'A']--;
                start++;
            }
            maxSize = Math.max(maxSize, (end - start + 1));
        }
        return maxSize;
    }

    // 133. Clone Graph

    // Given a reference of a node in a connected undirected graph.
    // Return a deep copy (clone) of the graph.
    // Each node in the graph contains a value (int) and a list (List[Node]) of its neighbors.

    // class Node {
    //     public int val;
    //     public List<Node> neighbors;
    // }
    
    // Test case format:
    // For simplicity, each node's value is the same as the node's index (1-indexed). For example, the first node with val == 1, the second node with val == 2, and so on. The graph is represented in the test case using an adjacency list.
    // An adjacency list is a collection of unordered lists used to represent a finite graph. Each list describes the set of neighbors of a node in the graph.
    // The given node will always be the first node with val = 1. You must return the copy of the given node as a reference to the cloned graph.

    /*
    // Definition for a Node.
    class Node {
        public int val;
        public List<Node> neighbors;
        public Node() {
            val = 0;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val) {
            val = _val;
            neighbors = new ArrayList<Node>();
        }
        public Node(int _val, ArrayList<Node> _neighbors) {
            val = _val;
            neighbors = _neighbors;
        }
    }
    */
    private Map<Node, Node> visited = new HashMap<>();
    public Node cloneGraph(Node node) {
        if (node == null)
            return node;

        // already visited before, then return clone from the visited
        if (visited.containsKey(node))
            return visited.get(node);

        // create a clone for given node and iterate to clone its neighbors.
        Node cloneNode = new Node(node.val, new ArrayList<>());
        visited.put(node, cloneNode);
        for (Node neigh : node.neighbors) {
            cloneNode.neighbors.add(cloneGraph(neigh));
        }
        return cloneNode;
    }

    // 96. Unique Binary Search Trees

    // Given an integer n, return the number of structurally unique BST's (binary search trees) which has exactly n nodes of unique values from 1 to n.

    // Example 1:

    // Input: n = 3
    // Output: 5

    public int numTrees(int n) {
        // numTree[root] = numTree[left] * numTree[right]
        // numTree[4] = numTree[0] * numTree[3] +
        // numTree[1] * numTree[2] +
        // numTree[2] * numTree[1] +
        // numTree[3] * numTree[1]

        int [] G = new int[n+1];
        G[0] = G[1] = 1;
        for(int i = 2; i <= n; ++i) {
            int total = 0;
            for(int j = 1; j <= i; ++j) {
                int left = j - 1; // root - 1
                int right = i - j; // node - root
                G[i] += G[left] * G[right];
            }
        }
        return G[n];
    }

    // 54. Spiral Matrix
    // Given an m x n matrix, return all elements of the matrix in spiral order.
    // Example 1:

    // Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    // Output: [1,2,3,6,9,8,7,4,5]

    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<>();
        int rows = matrix.length;
        int columns = matrix[0].length;
        int up = 0;
        int down = rows - 1;
        int left = 0;
        int right = columns - 1;

        while (result.size() < rows * columns) {
            // Traverse from left to right.
            for (int col = left; col <= right; col++) {
                result.add(matrix[up][col]);
            }

            // Traverse downwards.
            for (int row = up + 1; row <= down; row++) {
                result.add(matrix[row][right]);
            }

            // Make sure we are now on a different row.
            if (up != down) {
                // Traverse from right to left.
                for (int col = right - 1; col >= left; col--) {
                    result.add(matrix[down][col]);
                }
            }
            // Make sure we are now on a different column.
            if (left != right) {
                // Traverse upwards.
                for (int row = down - 1; row > up; row--) {
                    result.add(matrix[row][left]);
                }
            }
            left++;
            right--;
            up++;
            down--;
        }
        return result;
    }

    // 238. Product of Array Except Self

    // Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].

    // The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.

    // You must write an algorithm that runs in O(n) time and without using the division operation.

    // Example 1:

    // Input: nums = [1,2,3,4]
    // Output: [24,12,8,6]

    public int[] productExceptSelf(int[] nums) {
        int length = nums.length;
        int[] left = new int[length];
        int[] right = new int[length];
        int[] answer = new int[length];

        left[0] = 1;
        for(int i = 1; i < length; i++) {
            left[i] = left[i - 1] * nums[i - 1];
        }

        right[length - 1] = 1;
        for(int j = length - 2; j >= 0; j--) {
            right[j] = right[j + 1] * nums[j + 1];
        }

        for(int k = 0; k < length; k++) {
            answer[k] = left[k] * right[k];
        }
        return answer;
    }

    // 56. Merge Intervals

    // Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.

    // Example 1:

    // Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    // Output: [[1,6],[8,10],[15,18]]
    // Explanation: Since intervals [1,3] and [2,6] overlap, merge them into [1,6].

    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (a, b) -> Integer.compare(a[0], b[0]));
        LinkedList<int[]> merged = new LinkedList<>();
        for (int[] interval : intervals) {
            // if the list of merged intervals is empty or if the current
            // interval does not overlap with the previous, simply append it.
            if (merged.isEmpty() || merged.getLast()[1] < interval[0]) {
                merged.add(interval);
            }
            // otherwise, there is overlap, so we merge the current and previous
            // intervals.
            else {
                merged.getLast()[1] = Math.max(merged.getLast()[1], interval[1]);
            }
        }
        return merged.toArray(new int[merged.size()][]);
    }

    // 213. House Robber II

    // You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have a security system connected, and it will automatically contact the police if two adjacent houses were broken into on the same night.

    // Given an integer array nums representing the amount of money of each house, return the maximum amount of money you can rob tonight without alerting the police.

    // Example 1:

    // Input: nums = [2,3,2]
    // Output: 3
    // Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2), because they are adjacent houses.

    public int rob(int[] nums) {
        int n = nums.length;
        // Array from index 0 to n-2 (inclusive)
        int[] array1 = new int[n -1];
        System.arraycopy(nums, 0, array1, 0, n-1);
        // Array from index 1 to n-1 (inclusive)
        int[] array2 = new int[n -1];
        System.arraycopy(nums, 1, array2, 0, n-1);

        int result = Math.max(nums[0], Math.max(rob1(array1), rob1(array2)));
        return result;

    }

    private int rob1(int[] nums) {
        int rob1 = 0;
        int rob2 = 0;
        for(int n: nums) {
            int temp = Math.max((rob1 + n), rob2);
            rob1 = rob2;
            rob2 = temp;
        }
        return rob2;
    }

    // 19. Remove Nth Node From End of List
    // Given the head of a linked list, remove the nth node from the end of the list and return its head.

    // Example 1:

    // Input: head = [1,2,3,4,5], n = 2
    // Output: [1,2,3,5]

    public class ListNode {
        int val;
        ListNode next;
        ListNode() {}
        ListNode(int val) { this.val = val; }
        ListNode(int val, ListNode next) { this.val = val; this.next = next; }
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode result = new ListNode(-1);
        result.next = head;
        ListNode start = result;
        ListNode end = result.next;
        while(end != null && n > 0) {
            end = end.next;
            n--;
        }

        while(end != null) {
            start = start.next;
            end = end.next;
        }
        start.next = start.next.next;
        return result.next;
    }

    // 138. Copy List with Random Pointer

    // A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
    // Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.
    // For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.
    // Return the head of the copied linked list.
    // The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

    // val: an integer representing Node.val
    // random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
    // Your code will only be given the head of the original linked list.

    // Example 1:

    // Input: head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
    // Output: [[7,null],[13,0],[11,4],[10,2],[1,0]]

    class Solution {
        class Node {
            int val;
            Node next;
            Node random;

            public Node(int val) {
                this.val = val;
                this.next = null;
                this.random = null;
            }
        }
        private Map<Node, Node> map = new HashMap<>();
        public Node copyRandomList(Node head) {
            if (head == null)
                return null;

            Node curr = head;
            while(curr != null) {
                Node copy = new Node(curr.val);
                map.put(curr, copy);
                curr = curr.next;
            }

            curr = head;
            while(curr != null) {
                Node copy = map.get(curr);
                copy.next = map.get(curr.next);
                copy.random = map.get(curr.random);
                map.put(curr, copy);
                curr = curr.next;
            }

            return map.get(head);
        }
    }

    // 221. Maximal Square

    // Given an m x n binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

    // Example 1:

    // Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]] 2x2 = 4
    // Output: 4

    public int maximalSquare(char[][] matrix) {
        int dp[][] = new int[matrix.length][matrix[0].length];
        int ans = 0;
        for (int i = 0; i < dp.length; i++) {
            if (matrix[i][0] == '1') {
                dp[i][0] = 1;
                ans = 1;
            }
        }
        for (int j = 0; j < dp[0].length; j++) {
            if (matrix[0][j] == '1') {
                dp[0][j] = 1;
                ans = 1;
            }
        }
        for (int i = 1; i < dp.length; i++) {
            for (int j = 1; j < dp[0].length; j++) {
                if (matrix[i][j] == '1') {
                    dp[i][j] = Math.min(dp[i - 1][j - 1], Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
                    ans = Math.max(dp[i][j], ans);
                }
            }
        }
        return ans * ans;
    }

    // 91. Decode Ways
    // A message containing letters from A-Z can be encoded into numbers using the following mapping:

    // 'A' -> "1"
    // 'B' -> "2"
    // ...
    // 'Z' -> "26"
    // To decode an encoded message, all the digits must be grouped then mapped back into letters using the reverse of the mapping above (there may be multiple ways). For example, "11106" can be mapped into:

    // "AAJF" with the grouping (1 1 10 6)
    // "KJF" with the grouping (11 10 6)
    // Note that the grouping (1 11 06) is invalid because "06" cannot be mapped into 'F' since "6" is different from "06".

    // Given a string s containing only digits, return the number of ways to decode it.

    // The test cases are generated so that the answer fits in a 32-bit integer.

    

    // Example 1:

    // Input: s = "12"
    // Output: 2
    // Explanation: "12" could be decoded as "AB" (1 2) or "L" (12).
    // Example 2:

    // Input: s = "226"
    // Output: 3
    // Explanation: "226" could be decoded as "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6).

    public int numDecodings(String s) {
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for(int i = 2; i <= s.length(); i++) {
            int oneDigit = Integer.valueOf(s.substring(i - 1, i));
            int twoDigits = Integer.valueOf(s.substring(i - 2, i));
            if(oneDigit >= 1) {
                dp[i] += dp[i - 1];
            }
            if(twoDigits >= 10 && twoDigits <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[s.length()];
    }

    // 62. Unique Paths

    // There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

    // Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

    // The test cases are generated so that the answer will be less than or equal to 2 * 109.

    // Example 1:

    // Input: m = 3, n = 7
    // Output: 28

    // Example 2:

    // Input: m = 3, n = 2
    // Output: 3
    // Explanation: From the top-left corner, there are a total of 3 ways to reach the bottom-right corner:
    // 1. Right -> Down -> Down
    // 2. Down -> Down -> Right
    // 3. Down -> Right -> Down

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];

        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j -1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    // 323. Number of Connected Components in an Undirected graph

    // You have a graph of n nodes. You are given an integer n and an array edges
    // where edges[i] = [ai, bi] indicates that there is an edge between ai and bi
    // in the graph.

    // Return the number of connected components in the graph.

    // Example 1:
    // Input: n = 5, edges = [[0,1],[1,2],[3,4]]
    // Output: 2

    // Example 2:
    // Input: n = 5, edges = [[0,1],[1,2],[2,3],[3,4]]
    // Output: 1

    public int countComponents(int n, int[][] edges) {
        int components = 0;
        int[] visited = new int[n];
        List<Integer>[] adjList = new ArrayList[n];
        for (int i = 0; i < n; i++) {
            adjList[i] = new ArrayList<>();
        }

        for (int i = 0; i < edges.length; i++) {
            adjList[edges[i][0]].add(edges[i][1]);
            adjList[edges[i][1]].add(edges[i][0]);
        }

        for (int i = 0; i < n; i++) {
            if (visited[i] == 0) {
                components++;
                dfsCountComponents(adjList, visited, i);
            }
        }

        return components;
    }

    private void dfsCountComponents(List<Integer>[] adjList, int[] visited, int startNode) {
        visited[startNode] = 1;
        for (int i = 0; i < adjList[startNode].size(); i++) {
            if (visited[adjList[startNode].get(i)] == 0)
                dfsCountComponents(adjList, visited, adjList[startNode].get(i));
        }
    }

    // 287. Find the Duplicate Number

    // Explaination: https://www.youtube.com/watch?v=wjYnzkAhcNk&list=PLot-Xpze53lfOdF3KwpMSFEyfE77zIwiP&index=42

    // Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.

    // There is only one repeated number in nums, return this repeated number.

    // You must solve the problem without modifying the array nums and uses only constant extra space.

    // Example 1:

    // Input: nums = [1,3,4,2,2]
    // Output: 2
    // Example 2:

    // Input: nums = [3,1,3,4,2]
    // Output: 3

    public int findDuplicate(int[] nums) {
        if(nums.length > 1){
            int slow= nums[0];
            int fast = nums[nums[0]];
            while(slow != fast){
                slow = nums[slow];
                fast = nums[nums[fast]];
            }
            fast = 0;
            while(fast != slow){
                fast = nums[fast];
                slow = nums[slow];
            }
            return slow;
        }
        return -1;
    }

    // 253. Meeting Rooms II

    // Given an array of meeting time intervals intervals where intervals[i] =
    // [starti, endi], return the minimum number of conference rooms required.

    // Example 1:
    // Input: intervals = [[0,30],[5,10],[15,20]]
    // Output: 2

    // Example 2:
    // Input: intervals = [[7,10],[2,4]]
    // Output: 1

    public int minMeetingRooms(int[][] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;

        Arrays.sort(intervals, (a, b) -> a[0] - b[0]);
        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> a[1] - b[1]);
        minHeap.add(intervals[0]);

        for (int i = 1; i < intervals.length; i++) {
            int[] current = intervals[i];
            int[] earliest = minHeap.remove();

            if (current[0] >= earliest[1])
                earliest[1] = current[1];
            else
                minHeap.add(current);
            minHeap.add(earliest);
        }

        return minHeap.size();
    }

    // OR

    public int minMeetingRooms(int[][] intervals) {
        int[] start = new int[intervals.length];
        int[] end = new int[intervals.length];

        for (int i = 0; i < intervals.length; i++) {
            start[i] = intervals[i][0];
            end[i] = intervals[i][1];
        }

        int s = 0;
        int e = 0;
        int res = 0;
        int count = 0;
        while(s < intervals.length) {
            if(start[s] < end[e]) {
                s += 1;
                count += 1;
            } else {
                e += 1;
                count -=1;
            }
            count = Math.max(res, count);
        }
        return res;
    }

    // 1041. Robot Bounded In Circle

    // On an infinite plane, a robot initially stands at (0, 0) and faces north. Note that:

    // The north direction is the positive direction of the y-axis.
    // The south direction is the negative direction of the y-axis.
    // The east direction is the positive direction of the x-axis.
    // The west direction is the negative direction of the x-axis.
    // The robot can receive one of three instructions:

    // "G": go straight 1 unit.
    // "L": turn 90 degrees to the left (i.e., anti-clockwise direction).
    // "R": turn 90 degrees to the right (i.e., clockwise direction).
    // The robot performs the instructions given in order, and repeats them forever.

    // Return true if and only if there exists a circle in the plane such that the robot never leaves the circle.

    // Example 1:

    // Input: instructions = "GGLLGG"
    // Output: true
    // Explanation: The robot is initially at (0, 0) facing the north direction.
    // "G": move one step. Position: (0, 1). Direction: North.
    // "G": move one step. Position: (0, 2). Direction: North.
    // "L": turn 90 degrees anti-clockwise. Position: (0, 2). Direction: West.
    // "L": turn 90 degrees anti-clockwise. Position: (0, 2). Direction: South.
    // "G": move one step. Position: (0, 1). Direction: South.
    // "G": move one step. Position: (0, 0). Direction: South.
    // Repeating the instructions, the robot goes into the cycle: (0, 0) --> (0, 1) --> (0, 2) --> (0, 1) --> (0, 0).
    // Based on that, we return true.
    // Example 2:

    // Input: instructions = "GG"
    // Output: false
    // Explanation: The robot is initially at (0, 0) facing the north direction.
    // "G": move one step. Position: (0, 1). Direction: North.
    // "G": move one step. Position: (0, 2). Direction: North.
    // Repeating the instructions, keeps advancing in the north direction and does not go into cycles.
    // Based on that, we return false.

    public boolean isRobotBounded(String ins) {
        int x = 0;
        int y = 0;
        int direction = 0;
        int d[][] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
        for (char ch: ins.toCharArray()) {
            if (ch == 'R')
                direction = (direction + 1) % 4;
            else if (ch == 'L')
                direction = (direction + 3) % 4;
            else {
                x += d[direction][0]; 
                y += d[direction][1];
            }
        }
        return x == 0 && y == 0 || direction > 0;
    }

    // 230. Kth Smallest Element in a BST

    // Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.

    // Example 1:

    // Input: root = [3,1,4,null,2], k = 1
    // Output: 1

    public int kthSmallest(TreeNode root, int k) {
        List<Integer> nums = inorder(root, new ArrayList<Integer>());
        return nums.get(k - 1);
    }

    private List<Integer> inorder(TreeNode root, ArrayList<Integer> arr) {
        if(root == null) {
            return arr;
        }
        inorder(root.left, arr);
        arr.add(root.val);
        inorder(root.right, arr);
        return arr;
    }

    // 647. Palindromic Substrings

    // Given a string s, return the number of palindromic substrings in it.

    // A string is a palindrome when it reads the same backward as forward.

    // A substring is a contiguous sequence of characters within the string.

    

    // Example 1:

    // Input: s = "abc"
    // Output: 3
    // Explanation: Three palindromic strings: "a", "b", "c".
    // Example 2:

    // Input: s = "aaa"
    // Output: 6
    // Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".

    public int countSubstrings(String s) {
        int res = 0;
        for(int i = 0; i < s.length(); i++) {
            // Odd
            int l = i;
            int r = i;
            while(l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
                res += 1;
                l -= 1;
                r += 1;
            }

            // Even
            l = i;
            r = i + 1;
            while(l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
                res += 1;
                l -= 1;
                r += 1;
            }
        }
        return res;
    }

    // 417. Pacific Atlantic Water Flow

    // There is an m x n rectangular island that borders both the Pacific Ocean and Atlantic Ocean. The Pacific Ocean touches the island's left and top edges, and the Atlantic Ocean touches the island's right and bottom edges.

    // The island is partitioned into a grid of square cells. You are given an m x n integer matrix heights where heights[r][c] represents the height above sea level of the cell at coordinate (r, c).

    // The island receives a lot of rain, and the rain water can flow to neighboring cells directly north, south, east, and west if the neighboring cell's height is less than or equal to the current cell's height. Water can flow from any cell adjacent to an ocean into the ocean.

    // Return a 2D list of grid coordinates result where result[i] = [ri, ci] denotes that rain water can flow from cell (ri, ci) to both the Pacific and Atlantic oceans.

    // Example 1:

    // Input: heights = [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    // Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]
    // Explanation: The following cells can flow to the Pacific and Atlantic oceans, as shown below:
    // [0,4]: [0,4] -> Pacific Ocean 
    //     [0,4] -> Atlantic Ocean
    // [1,3]: [1,3] -> [0,3] -> Pacific Ocean 
    //     [1,3] -> [1,4] -> Atlantic Ocean
    // [1,4]: [1,4] -> [1,3] -> [0,3] -> Pacific Ocean 
    //     [1,4] -> Atlantic Ocean
    // [2,2]: [2,2] -> [1,2] -> [0,2] -> Pacific Ocean 
    //     [2,2] -> [2,3] -> [2,4] -> Atlantic Ocean
    // [3,0]: [3,0] -> Pacific Ocean 
    //     [3,0] -> [4,0] -> Atlantic Ocean
    // [3,1]: [3,1] -> [3,0] -> Pacific Ocean 
    //     [3,1] -> [4,1] -> Atlantic Ocean
    // [4,0]: [4,0] -> Pacific Ocean 
    //     [4,0] -> Atlantic Ocean
    // Note that there are other possible paths for these cells to flow to the Pacific and Atlantic oceans.

    int dir[][] = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

    public List<List<Integer>> pacificAtlantic(int[][] matrix) {
        List<List<Integer>> res = new ArrayList<>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return res;

        int row = matrix.length, col = matrix[0].length;
        // store cells that reach to both atlantic and pacific oceans
        boolean[][] pacific = new boolean[row][col];
        boolean[][] atlantic = new boolean[row][col];

        // DFS
        for (int i = 0; i < col; i++) {
            dfsPacificAtlantic(matrix, 0, i, Integer.MIN_VALUE, pacific);
            dfsPacificAtlantic(matrix, row - 1, i, Integer.MIN_VALUE, atlantic);
        }

        for (int i = 0; i < row; i++) {
            dfsPacificAtlantic(matrix, i, 0, Integer.MIN_VALUE, pacific);
            dfsPacificAtlantic(matrix, i, col - 1, Integer.MIN_VALUE, atlantic);
        }

        // Find the cells and add to result if both pacific and atlantic cells are true.
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                if (pacific[i][j] && atlantic[i][j])
                    res.add(Arrays.asList(i, j));

        return res;
    }

    private void dfsPacificAtlantic(int[][] matrix, int i, int j, int prev, boolean[][] ocean) {
        if (i < 0 || i >= ocean.length || j < 0 || j >= ocean[0].length)
            return;

        if (matrix[i][j] < prev || ocean[i][j])
            return;

        ocean[i][j] = true;
        for (int[] d : dir) {
            dfsPacificAtlantic(matrix, i + d[0], j + d[1], matrix[i][j], ocean);
        }
    }

    // 208. Implement Trie (Prefix Tree)

    Implement the Trie class:

    // Trie() Initializes the trie object.
    // void insert(String word) Inserts the string word into the trie.
    // boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.
    // boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.
    
    // Example 1:

    // Input
    // ["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
    // [[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
    // Output
    // [null, null, true, false, true, null, true]

    // Explanation
    // Trie trie = new Trie();
    // trie.insert("apple");
    // trie.search("apple");   // return True
    // trie.search("app");     // return False
    // trie.startsWith("app"); // return True
    // trie.insert("app");
    // trie.search("app");     // return True

    class TrieNode {
        public char val;
        public boolean endOfWord;
        public TrieNode[] children = new TrieNode[26];

        public TrieNode() {
        }

        TrieNode(char c) {
            TrieNode node = new TrieNode();
            node.val = c;
        }
    }

    class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
            root.val = ' ';
        }
        
        public void insert(String word) {
            TrieNode node = root;
            for(char ch: word.toCharArray()) {
                if(node.children[ch - 'a'] == null) {
                    node.children[ch - 'a'] = new TrieNode(ch);
                }
                node = node.children[ch - 'a'];
            }
            node.endOfWord = true;
        }
        
        public boolean search(String word) {
            TrieNode node = root;
            for(char ch: word.toCharArray()) {
                if(node.children[ch - 'a'] == null) {
                    return false;
                }
                node = node.children[ch - 'a'];
            }
            return node.endOfWord;
        }
        
        public boolean startsWith(String prefix) {
            TrieNode node = root;
            for(char ch: prefix.toCharArray()) {
                if(node.children[ch - 'a'] == null) {
                    return false;
                }
                node = node.children[ch - 'a'];
            }
            return true;
        }
    }

    // 36. Valid Sudoku

    // Determine if a 9 x 9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

    // Each row must contain the digits 1-9 without repetition.
    // Each column must contain the digits 1-9 without repetition.
    // Each of the nine 3 x 3 sub-boxes of the grid must contain the digits 1-9 without repetition.
    // Note:

    // A Sudoku board (partially filled) could be valid but is not necessarily solvable.
    // Only the filled cells need to be validated according to the mentioned rules.

    // Example 1:

    // Input: board = 
    // [["5","3",".",".","7",".",".",".","."]
    // ,["6",".",".","1","9","5",".",".","."]
    // ,[".","9","8",".",".",".",".","6","."]
    // ,["8",".",".",".","6",".",".",".","3"]
    // ,["4",".",".","8",".","3",".",".","1"]
    // ,["7",".",".",".","2",".",".",".","6"]
    // ,[".","6",".",".",".",".","2","8","."]
    // ,[".",".",".","4","1","9",".",".","5"]
    // ,[".",".",".",".","8",".",".","7","9"]]
    // Output: true

    public boolean isValidSudoku(char[][] board) {
        Set seen = new HashSet();
        for (int i=0; i<9; ++i) {
            for (int j=0; j<9; ++j) {
                char number = board[i][j];
                if (number != '.')
                    if (!seen.add(number + " in row " + i) ||
                        !seen.add(number + " in column " + j) ||
                        !seen.add(number + " in block " + i/3 + "-" + j/3))
                        return false;
            }
        }
        return true;
    }

    // 560. Subarray Sum Equals K

    // Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.

    // A subarray is a contiguous non-empty sequence of elements within an array.

    // Example 1:

    // Input: nums = [1,1,1], k = 2
    // Output: 2
    // Example 2:

    // Input: nums = [1,2,3], k = 3
    // Output: 2

    public int subarraySum(int[] nums, int k) {
        Map<Integer, Integer> prefixSumCount = new HashMap<>();
        int result = 0;
        int currSum = 0;
        prefixSumCount.put(0, 1);
        for(int n: nums) {
            currSum += n;
            int diff = currSum - k;
            result += prefixSumCount.getOrDefault(diff, 0);
            prefixSumCount.put(currSum, 1 + prefixSumCount.getOrDefault(currSum, 0));
        }
        return result;
    }

    // 43. Multiply Strings

    // Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

    // Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.

    // Example 1:

    // Input: num1 = "2", num2 = "3"
    // Output: "6"
    // Example 2:

    // Input: num1 = "123", num2 = "456"
    // Output: "56088"

    public String multiply(String num1, String num2) {
        if(num1.equals("0") || num2.equals("0") || num1 == null || num2 == null) {
            return "0";
        }
        int m = num1.length();
        int n = num2.length();
        int[] pos = new int[m + n];

        for(int i = m - 1; i >= 0; i--) {
            for(int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j;
                int p2 = i + j + 1;
                int sum = mul + pos[p2];

                pos[p1] += sum / 10;
                pos[p2] = sum % 10; 
            }
        }

        StringBuilder sb = new StringBuilder();
        for(int p: pos) {
            if(!(sb.length() == 0 && p == 0)) {
                sb.append(p);
            }
        }
        return sb.length() == 0 ? "0" : sb.toString();
    }

    // 1466. Reorder Routes to Make All Paths Lead to the City Zero

    // There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way to travel between two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in one direction because they are too narrow.

    // Roads are represented by connections where connections[i] = [ai, bi] represents a road from city ai to city bi.

    // This year, there will be a big event in the capital (city 0), and many people want to travel to this city.

    // Your task consists of reorienting some roads such that each city can visit the city 0. Return the minimum number of edges changed.

    // It's guaranteed that each city can reach city 0 after reorder.

    // Example 1:

    // Input: n = 6, connections = [[0,1],[1,3],[2,3],[4,0],[4,5]]
    // Output: 3
    // Explanation: Change the direction of edges show in red such that each node can reach the node 0 (capital).

    public int minReorder(int n, int[][] connections) {
        // start at city 0
        // recursively check its neighbors
        // count outgoing edges
        LinkedList<int[]>[] g = new LinkedList[n];
        for(int i = 0; i != n; ++i) {
            g[i] = new LinkedList<>(); //create graph  
        }

        for(int[] c: connections) { //put all edges 
            g[c[0]].add(new int[]{c[1], 1}); //index[1] == 1 - road is present
            g[c[1]].add(new int[]{c[0], 0}); //index[1] == 0 - road is absent
        }  

        int[] vis = new int[n];
        int reorderRoads = 0;
        LinkedList<Integer> q = new LinkedList<>(); //queue for BFS
        q.add(0);
        while(! q.isEmpty()) {
            int city = q.poll();
            if(vis[city] == 1)
                continue;
            vis[city] = 1;
            for(int[] neib: g[city]) {
                if(vis[neib[0]] == 0){
                    q.add(neib[0]);
                    if(neib[1] == 1) {
                        ++reorderRoads;
                    }
                }
            }
        }
        return reorderRoads;
    }

    // 7. Interleaving String

    // Given strings s1, s2, and s3, find whether s3 is formed by an interleaving of s1 and s2.

    // An interleaving of two strings s and t is a configuration where s and t are divided into n and m substrings respectively, such that:

    // s = s1 + s2 + ... + sn
    // t = t1 + t2 + ... + tm
    // |n - m| <= 1
    // The interleaving is s1 + t1 + s2 + t2 + s3 + t3 + ... or t1 + s1 + t2 + s2 + t3 + s3 + ...
    // Note: a + b is the concatenation of strings a and b.




    // 49. Group Anagrams

    // Given an array of strings strs, group the anagrams together. You can return the answer in any order.

    // An Anagram is a word or phrase formed by rearranging the letters of a different word or phrase, typically using all the original letters exactly once.

    // Example 1:

    // Input: strs = ["eat","tea","tan","ate","nat","bat"]
    // Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

    public List<List<String>> groupAnagrams(String[] strs) {
        List<List<String>> result = new ArrayList<>();
        Map<String, List<String>> map = new HashMap<>();
        for (String s : strs) {
            char[] chars = s.toCharArray();
            Arrays.sort(chars);
            String current = new String(chars);
            if (!map.containsKey(current)) {
                map.put(current, new ArrayList<>());
            }
            map.get(current).add(s);
        }
        result.addAll(map.values());
        return result;
    }

    // 153. Find Minimum in Rotated Sorted Array

    // Suppose an array of length n sorted in ascending order is rotated between 1 and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    // [4,5,6,7,0,1,2] if it was rotated 4 times.
    // [0,1,2,4,5,6,7] if it was rotated 7 times.
    // Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

    // Given the sorted rotated array nums of unique elements, return the minimum element of this array.

    // You must write an algorithm that runs in O(log n) time.

    // Example 1:

    // Input: nums = [3,4,5,1,2]
    // Output: 1
    // Explanation: The original array was [1,2,3,4,5] rotated 3 times.

    public int findMin(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        if(nums.length == 0) return -1;
        if(nums.length == 1) return nums[0];
        if(nums[right] > nums[left]) {
            return nums[left];
        }
        while(right >= left) {
            int mid = left + (right - left) / 2;
            if(nums[mid] > nums[mid + 1]) {
                return nums[mid + 1];
            }
            if(nums[mid] < nums[mid - 1]) {
                return nums[mid];
            }
            if(nums[mid] > nums[0]) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }

    // 102. Binary Tree Level Order Traversal
    // Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

    // Example 1:

    // Input: root = [3,9,20,null,null,15,7]
    // Output: [[3],[9,20],[15,7]]

    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> levelOrder = new ArrayList<>();
        if (root == null)
            return levelOrder;

        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            List<Integer> currentLevel = new ArrayList<>();
            int size = queue.size();
            for (int i = 0; i < size; i++) {
                TreeNode node = queue.remove();
                currentLevel.add(node.val);
                if (node.left != null)
                    queue.add(node.left);
                if (node.right != null)
                    queue.add(node.right);
            }
            levelOrder.add(currentLevel);
        }
        return levelOrder;
    }

    // 1963. Minimum Number of Swaps to Make the String Balanced

    // You are given a 0-indexed string s of even length n. The string consists of exactly n / 2 opening brackets '[' and n / 2 closing brackets ']'.

    // A string is called balanced if and only if:

    // It is the empty string, or
    // It can be written as AB, where both A and B are balanced strings, or
    // It can be written as [C], where C is a balanced string.
    // You may swap the brackets at any two indices any number of times.

    // Return the minimum number of swaps to make s balanced.

    // Example 1:

    // Input: s = "][]["
    // Output: 1
    // Explanation: You can make the string balanced by swapping index 0 with index 3.
    // The resulting string is "[[]]".

    public int minSwaps(String s) {
        int closeCount = 0;
        int maxClose = 0;
        for(char ch: s.toCharArray()) {
            if(ch == '[') {
                closeCount -= 1;
            } else {
                closeCount += 1;
            }
            maxClose = Math.max(maxClose, closeCount);
        }
        return (maxClose + 1) / 2;
    }

    // 338. Counting Bits

    // Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

    // Example 1:

    // Input: n = 2
    // Output: [0,1,1]
    // Explanation:
    // 0 --> 0
    // 1 --> 1
    // 2 --> 10

    public int[] countBits(int n) {
        int[] dp = new int[n + 1];
        java.util.Arrays.fill(dp, 0);
        int offset = 1;
        for(int i = 1; i <= n; i++) {
            if(offset * 2 == i) {
                offset = i;
            }
            dp[i] = 1 + dp[i - offset];
        }
        return dp;
    }

    // 120. Triangle

    // Given a triangle array, return the minimum path sum from top to bottom.

    // For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

    // Example 1:

    // Input: triangle = [[2],[3,4],[6,5,7],[4,1,8,3]]
    // Output: 11
    // Explanation: The triangle looks like:
    //    2
    //   3 4
    //  6 5 7
    // 4 1 8 3
    // The minimum path sum from top to bottom is 2 + 3 + 5 + 1 = 11 (underlined above).

    public int minimumTotal(List<List<Integer>> triangle) {
        int n = triangle.size();
        int[] dp = new int[n + 1];
        for(int level = n - 1; level >= 0; level--) {
            for(int i = 0; i <= level; i++) {
                dp[i] = triangle.get(level).get(i) + Math.min(dp[i], dp[i + 1]);
            }
        }
        return dp[0];
    }

    // 347. Top K Frequent Elements

    // Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

    // Example 1:

    // Input: nums = [1,1,1,2,2,3], k = 2
    // Output: [1,2]

    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> count = new HashMap<>();
        for(int num: nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }
        PriorityQueue<Integer> heap = new PriorityQueue<>((a, b) -> count.get(a) - count.get(b));
        for(int n: count.keySet()) {
            heap.add(n);
            if(heap.size() > k) {
                heap.poll();
            }
        }

        int[] top = new int[k];
        while(k-- > 0) {
            top[k] = heap.poll();
        }
        return top;
    }

    // OR // Similar Bucket sort implemention. // O(n)


    public int[] topKFrequent1(int[] nums, int k) {
        Map<Integer, Integer> countFrequency = new HashMap<>();
        List<Integer>[] frequency = new ArrayList[nums.length + 1];

        // Initialize each list in the array
        for(int i = 0; i <= nums.length; i++){
            frequency[i] = new ArrayList<>();
        }

        for(int num: nums) {
            countFrequency.put(num, countFrequency.getOrDefault(num, 0) + 1);
        }
        for (Map.Entry<Integer, Integer> entry: countFrequency.entrySet()) {
            int count = entry.getKey();
            int value = entry.getValue();
            frequency[value].add(count);
        }

        int ans[] = new int[k];
        int idx = 0;
        for(int i = frequency.length; i >= 0 && idx < k; i--) {
            List<Integer> current = frequency[i - 1];
            if(!current.isEmpty()) {
                for(int e: current) {
                    ans[idx++] = e;
                }
                if(idx >= k) {
                    break;
                }
            }
        }
        return ans;
    }

    // 215. Kth Largest Element in an Array

    // Given an integer array nums and an integer k, return the kth largest element in the array.

    // Note that it is the kth largest element in the sorted order, not the kth distinct element.

    // Can you solve it without sorting?

    // Example 1:

    // Input: nums = [3,2,1,5,6,4], k = 2
    // Output: 5

    public int findKthLargest(int[] nums, int k) {
        k = nums.length - k;
        return quickSelect(nums, 0, nums.length - 1, k);
    }

    private int quickSelect(int[] nums, int l, int r, int k) {
        int pivot = nums[r];
        int p = l;
        for(int i = l; i < r; i++) {
            if(nums[i] <= pivot) {
                int temp = nums[i];
                nums[i] = nums[p];
                nums[p] = temp;
                p += 1;
            }
        }
        nums[r] = nums[p];
        nums[p] = pivot;

        if(p > k) {
            return quickSelect(nums, l, p - 1, k);
        } else if(p < k) {
            return quickSelect(nums, p + 1, r, k);
        } else {
            return nums[p];
        }
    }

    // 1888. Minimum Number of Flips to Make the Binary String Alternating

    // You are given a binary string s. You are allowed to perform two types of operations on the string in any sequence:

    // Type-1: Remove the character at the start of the string s and append it to the end of the string.
    // Type-2: Pick any character in s and flip its value, i.e., if its value is '0' it becomes '1' and vice-versa.
    // Return the minimum number of type-2 operations you need to perform such that s becomes alternating.

    // The string is called alternating if no two adjacent characters are equal.

    // For example, the strings "010" and "1010" are alternating, while the string "0100" is not.
    

    // Example 1:

    // Input: s = "111000"
    // Output: 2
    // Explanation: Use the first operation two times to make s = "100011".
    // Then, use the second operation on the third and sixth elements to make s = "101010".

    public int minFlips(String s) {
        StringBuilder sb = new StringBuilder(s).append(s);

        StringBuilder alt1 = new StringBuilder();
        StringBuilder alt2 = new StringBuilder();

        for (int i = 0; i < sb.length(); i++) {
            if (i % 2 == 0) {
                alt1.append(0);
                alt2.append(1);
            }
            else {
                alt1.append(1);
                alt2.append(0);
            }
        }

        int diff1 = 0, diff2 = 0;
        int l = 0;

        int res = sb.length();

        for (int r = 0; r < sb.length(); r++) {
            if (alt1.charAt(r) != sb.charAt(r)) diff1++;
            if (alt2.charAt(r) != sb.charAt(r)) diff2++;

            if (r - l + 1 > s.length()) {
                if (alt1.charAt(l) != sb.charAt(l)) diff1--;
                if (alt2.charAt(l) != sb.charAt(l)) diff2--;
                l++;
            }

            if (r - l + 1 == s.length()) {
                res = Math.min(res, Math.min(diff1, diff2));
            }
        }

        return res;
    }

    // 34. Find First and Last Position of Element in Sorted Array

    // Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

    // If target is not found in the array, return [-1, -1].

    // You must write an algorithm with O(log n) runtime complexity.

    // Example 1:

    // Input: nums = [5,7,7,8,8,10], target = 8
    // Output: [3,4]

    // Example 2:

    // Input: nums = [5,7,7,8,8,10], target = 6
    // Output: [-1,-1]

    public int[] searchRange(int[] nums, int target) {
        int[] arr={-1,-1};
        arr[0]=findIndBinarySearch(nums,target,true);
        arr[1]=findIndBinarySearch(nums,target,false);
        return arr;
    }

    public int findIndBinarySearch(int[] nums,int target,boolean flag)
    {
        int left=0;
        int right=nums.length-1;
        int i = -1;

        while(left <= right) {
            int mid=(left + right)/2;

            if(nums[mid] > target) {
                right = mid - 1;
            } else if(nums[mid] < target) {
                left = mid + 1;
            } else if(nums[mid] == target) {
                i = mid;
                if(flag) {
                  right = mid - 1;
                } else {
                   left = mid + 1;
                }
            }
        }
        return i;
    }

    // 24. Swap Nodes in Pairs

    // Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

    // Example 1:

    // Input: head = [1,2,3,4]
    // Output: [2,1,4,3]

    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;

        ListNode prevNode = dummy;

        while ((head != null) && (head.next != null)) {

        // Nodes to be swapped
        ListNode firstNode = head;
        ListNode secondNode = head.next;

        // Swapping
        prevNode.next = secondNode;
        firstNode.next = secondNode.next;
        secondNode.next = firstNode;

        // Reinitializing the head and prevNode for next swap
        prevNode = firstNode;
        head = firstNode.next; // jump
        }

        // Return the new head node.
        return dummy.next;
    }

    // 416. Partition Equal Subset Sum

    // Given an integer array nums, return true if you can partition the array into two subsets such that the sum of the elements in both subsets is equal or false otherwise.

    // Example 1:

    // Input: nums = [1,5,11,5]
    // Output: true
    // Explanation: The array can be partitioned as [1, 5, 5] and [11].





    // 74. Search a 2D Matrix


    // You are given an m x n integer matrix matrix with the following two properties:

    // Each row is sorted in non-decreasing order.
    // The first integer of each row is greater than the last integer of the previous row.
    // Given an integer target, return true if target is in matrix or false otherwise.

    // You must write a solution in O(log(m * n)) time complexity.

    // Example 1:


    // Input: matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]], target = 3
    // Output: true


    public boolean searchMatrix(int[][] matrix, int target) {
        int ROWS = matrix.length;
        int COLS = matrix[0].length;
        int top = 0;
        int bottom = ROWS - 1;

        while(top <= bottom) {
            int row = (top + bottom) / 2;
            if(target > matrix[row][COLS - 1]) {
                top = row + 1;
            } else if(target < matrix[row][0]) {
                bottom = row - 1;
            } else {
                break;
            }
        }
        if(!(top <= bottom)) {
            return false;
        }

        int row = (top + bottom) / 2;
        int l = 0;
        int r = COLS - 1;
        while(l <= r) {
            int m = (l + r) / 2;
            if(target > matrix[row][m]) {
                l = m + 1;
            } else if (target < matrix[row][m]) {
                r = m - 1;
            } else {
                return true;
            }
        }
        return false;
    }

    // 1448. Count Good Nodes in Binary

    // Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a value greater than X.

    // Return the number of good nodes in the binary tree.

    // Example 1:

    // Input: root = [3,1,4,3,null,1,5]
    // Output: 4
    // Explanation: Nodes in blue are good.
    // Root Node (3) is always a good node.
    // Node 4 -> (3,4) is the maximum value in the path starting from the root.
    // Node 5 -> (3,4,5) is the maximum value in the path
    // Node 3 -> (3,1,3) is the maximum value in the path.

    int count = 0;
    public int goodNodes(TreeNode root) {
        int maximum = root.val;
        countGoodNodes(root,  maximum);
        return count;
    }

    public void countGoodNodes(TreeNode root, int maximum) {
        if(root == null) {
            return;
        }
        if(root!=null) {
            if (maximum <= root.val) {
                maximum = root.val;
                count++;
            }
        }
        countGoodNodes(root.left,maximum);
        countGoodNodes(root.right, maximum);
    }

    // 57. Insert Interval

    // You are given an array of non-overlapping intervals intervals where intervals[i] = [starti, endi] represent the start and the end of the ith interval and intervals is sorted in ascending order by starti. You are also given an interval newInterval = [start, end] that represents the start and end of another interval.

    // Insert newInterval into intervals such that intervals is still sorted in ascending order by starti and intervals still does not have any overlapping intervals (merge overlapping intervals if necessary).

    // Return intervals after the insertion.

    // Example 1:

    // Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    // Output: [[1,5],[6,9]]

    public int[][] insert(int[][] intervals, int[] newInterval) {
        // init data
        int newStart = newInterval[0];
        int newEnd = newInterval[1];
        int idx = 0;
        int n = intervals.length;
        LinkedList<int[]> output = new LinkedList<>();

        // Add all the intervals starting before newInterval
        while (idx < n && newStart > intervals[idx][0])
            output.add(intervals[idx++]);

        // Add newInterval
        int[] interval = new int[2];
        // if there is no overlap, just add the interval
        if (output.isEmpty() || output.getLast()[1] < newStart)
            output.add(newInterval);
        // if there is an overlap, merge with the last interval
        else {
            interval = output.removeLast();
            interval[1] = Math.max(interval[1], newEnd);
            output.add(interval);
        }

        // Add next intervals, merge with newInterval if needed
        while (idx < n) {
            interval = intervals[idx++];
            int start = interval[0], end = interval[1];
            // if there is no overlap, just add an interval
            if (output.getLast()[1] < start)
                output.add(interval);
            else {
                interval = output.removeLast();
                interval[1] = Math.max(interval[1], end);
                output.add(interval);
            }
        }
        return output.toArray(new int[output.size()][2]);
    }

    // 279. Perfect Squares

    // Given an integer n, return the least number of perfect square numbers that sum to n.

    // A perfect square is an integer that is the square of an integer; in other words, it is the product of some integer with itself. For example, 1, 4, 9, and 16 are perfect squares while 3 and 11 are not.
    
    // Example 1:

    // Input: n = 12
    // Output: 3
    // Explanation: 12 = 4 + 4 + 4.
    // Example 2:

    // Input: n = 13
    // Output: 2
    // Explanation: 13 = 4 + 9.


    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 0;
        
        for (int i = 1; i <= n; i++) {
            dp[i] = i;
            for (int j = 1; j * j <= i; j++) {
                int square = j * j;
                dp[i] = Math.min(dp[i], 1 + dp[i - square]);
            }
        }
        
        return dp[n];
    }

    // 189. Rotate Array

    // Given an integer array nums, rotate the array to the right by k steps, where k is non-negative.

    // Example 1:

    // Input: nums = [1,2,3,4,5,6,7], k = 3
    // Output: [5,6,7,1,2,3,4]
    // Explanation:
    // rotate 1 steps to the right: [7,1,2,3,4,5,6]
    // rotate 2 steps to the right: [6,7,1,2,3,4,5]
    // rotate 3 steps to the right: [5,6,7,1,2,3,4]

    public void rotate(int[] nums, int k) {
        k = k % nums.length;
        reverse(nums,0,nums.length-1);
        reverse(nums,0,k - 1);
        reverse(nums, k, nums.length - 1);
    }

    public static void reverse(int[] nums,int start,int end){
        while(start < end){
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

    // 261. Graph Valid Tree

    // You have a graph of n nodes labeled from 0 to n - 1. You are given an integer
    // n and a list of edges where edges[i] = [ai, bi] indicates that there is an
    // undirected edge between nodes ai and bi in the graph.

    // Return true if the edges of the given graph make up a valid tree, and false
    // otherwise.

    // Example 1:
    // Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
    // Output: true

    // Example 2:
    // Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
    // Output: false

    public boolean validTree(int n, int[][] edges) {
        if (edges.length != n - 1)
            return false;

        // Make the adjacency List.
        List<List<Integer>> adjList = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            adjList.add(new ArrayList<>());
        }

        for (int[] edge : edges) {
            adjList.get(edge[0]).add(edge[1]);
            adjList.get(edge[1]).add(edge[0]);
        }

        Queue<Integer> q = new LinkedList<>();
        Set<Integer> seen = new HashSet<>();
        q.offer(0);
        seen.add(0);

        while (!q.isEmpty()) {
            int current = q.poll();
            for (int neigh : adjList.get(current)) {
                if (seen.contains(neigh))
                    continue;
                seen.add(neigh);
                q.offer(neigh);
            }
        }

        return seen.size() == n;
    }

    // 743. Network Delay Time

    // You are given a network of n nodes, labeled from 1 to n. You are also given times, a list of travel times as directed edges times[i] = (ui, vi, wi), where ui is the source node, vi is the target node, and wi is the time it takes for a signal to travel from source to target.

    // We will send a signal from a given node k. Return the minimum time it takes for all the n nodes to receive the signal. If it is impossible for all the n nodes to receive the signal, return -1.
    
    // Example 1:

    // Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
    // Output: 2

    public int networkDelayTime(int[][] times, int n, int k) {
        // Create an adjacency list to represent the graph
        List<List<Pair<Integer, Integer>>> graph = new ArrayList<>();
        
        // Initialize the distance array to store minimum distances from the source node
        int[] dist = new int[n + 1];
        
        // Initialize the adjacency list for each node
        for (int i = 0; i < n + 1; i++) {
            graph.add(new ArrayList<>());
        }
        
        // Populate the adjacency list based on the given times array
        for (int[] time : times) {
            graph.get(time[0]).add(new Pair(time[1], time[2]));
        }
        
        // Initialize a priority queue for exploring nodes in order of their shortest distance
        PriorityQueue<Pair<Integer, Integer>> pq = new PriorityQueue<>((a, b) -> a.getValue() - b.getValue());
        
        // Add the starting node to the priority queue with distance 0
        pq.add(new Pair(k, 0));
        
        // Initialize all distances to infinity
        Arrays.fill(dist, Integer.MAX_VALUE);
        dist[k] = 0;
        
        // Explore nodes using Dijkstra's algorithm
        while (!pq.isEmpty()) {
            Pair<Integer, Integer> p = pq.poll();
            int cur = p.getKey();
            
            // Explore neighboring nodes and update distances
            for (Pair<Integer, Integer> neighbor : graph.get(cur)) {
                int curDist = neighbor.getValue() + dist[cur];
                
                // If the new distance is smaller, update and add to the priority queue
                if (curDist < dist[neighbor.getKey()]) {
                    dist[neighbor.getKey()] = curDist;
                    pq.add(new Pair(neighbor.getKey(), curDist));
                }
            }
        }
        
        // Find the maximum delay among all nodes
        int maxDelay = 0;
        for (int i = 1; i <= n; i++) {
            if (dist[i] == Integer.MAX_VALUE) {
                return -1; // Some nodes are unreachable, return -1
            }
            maxDelay = Math.max(maxDelay, dist[i]);
        }
        
        return maxDelay;
    }

    // 1911. Maximum Alternating Subsequence Sum

    // The alternating sum of a 0-indexed array is defined as the sum of the elements at even indices minus the sum of the elements at odd indices.

    // For example, the alternating sum of [4,2,5,3] is (4 + 5) - (2 + 3) = 4.
    // Given an array nums, return the maximum alternating sum of any subsequence of nums (after reindexing the elements of the subsequence).

    // A subsequence of an array is a new array generated from the original array by deleting some elements (possibly none) without changing the remaining elements' relative order. For example, [2,7,4] is a subsequence of [4,2,3,7,2,1,4] (the underlined elements), while [2,4,2] is not.

    // Example 1:

    // Input: nums = [4,2,5,3]
    // Output: 7
    // Explanation: It is optimal to choose the subsequence [4,2,5] with alternating sum (4 + 5) - 2 = 7.

    public long maxAlternatingSum(int[] nums) {
        int n = nums.length;
        long dp[][] = new long[n][2];
        dp[0][0] = nums[0];
        for(int i=1; i<n; i++){
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1]+nums[i]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0]-nums[i]);
        }

        return Math.max(dp[n-1][0], dp[n-1][1]);
    }

    // 337. House Robber III

    // The thief has found himself a new place for his thievery again. There is only one entrance to this area, called root.

    // Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that all houses in this place form a binary tree. It will automatically contact the police if two directly-linked houses were broken into on the same night.

    // Given the root of the binary tree, return the maximum amount of money the thief can rob without alerting the police.

    // Example 1:


    // Input: root = [3,2,3,null,3,null,1]
    // Output: 7
    // Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.

    public int rob(TreeNode root) {
        int[] num = dfsRob(root);
        return Math.max(num[0], num[1]);
    }

    private int[] dfsRob(TreeNode x) {
        if (x == null) 
            return new int[2];
        int[] left = dfsRob(x.left);
        int[] right = dfsRob(x.right);
        int[] res = new int[2];
        res[0] = left[1] + right[1] + x.val;
        res[1] = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return res;
    }

    // 394. Decode String

    // Given an encoded string, return its decoded string.

    // The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

    // You may assume that the input string is always valid; there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be input like 3a or 2[4].

    // The test cases are generated so that the length of the output will never exceed 105.

    

    // Example 1:

    // Input: s = "3[a]2[bc]"
    // Output: "aaabcbc"
    // Example 2:

    // Input: s = "3[a2[c]]"
    // Output: "accaccacc"


    public String decodeString(String s) {
        Stack<Integer> numStack = new Stack<>();
        Stack<StringBuilder> strBuild = new Stack<>();
        StringBuilder str = new StringBuilder();
        int num=0;
        for(char c: s.toCharArray()){
            if(c >= '0' && c <= '9'){
                num = num * 10 + c -'0';
            } else if (c=='[') {
                strBuild.push(str);
                str = new StringBuilder();
                numStack.push(num);
                num=0;
            } else if (c == ']') {
                StringBuilder temp = str;
                str = strBuild.pop();
                int count = numStack.pop();
                while (count-- > 0) {
                    str.append(temp);
                }
            } else {
                str.append(c);
            }
        }
        return str.toString();
    }

    // 1838. Frequency of the Most Frequent Element

    // The frequency of an element is the number of times it occurs in an array.

    // You are given an integer array nums and an integer k. In one operation, you can choose an index of nums and increment the element at that index by 1.

    // Return the maximum possible frequency of an element after performing at most k operations.

    // Example 1:

    // Input: nums = [1,2,4], k = 5
    // Output: 3
    // Explanation: Increment the first element three times and the second element two times to make nums = [4,4,4].
    // 4 has a frequency of 3.

    public int maxFrequency(int[] nums, int k) {
        int left = 0;
        int max = 0;
        int sum = 0;
        Arrays.sort(nums);
        for(int right = 0; right < nums.length; right++) {
            sum += nums[right];
            int windowLength = right - left + 1;
            while(nums[right] * windowLength > sum + k) {
                sum -= nums[left];
                left++;
            }
            max = Math.max(max, windowLength);
        }
        return max;
    }


    // 1871. Jump Game VII
    // You are given a 0-indexed binary string s and two integers minJump and maxJump. In the beginning, you are standing at index 0, which is equal to '0'. You can move from index i to index j if the following conditions are fulfilled: i + minJump <= j <= min(i + maxJump, s.length - 1), and s[j] == '0'.
    // Return true if you can reach index s.length - 1 in s, or false otherwise.

    // Example 1:

    // Input: s = "011010", minJump = 2, maxJump = 3
    // Output: true
    // Explanation:
    // In the first step, move from index 0 to index 3. 
    // In the second step, move from index 3 to index 5.

    // Example 2:

    // Input: s = "01101110", minJump = 2, maxJump = 3
    // Output: false

    public boolean canReach(String s, int minJump, int maxJump) {
        int n = s.length();
        Queue<Integer> q = new LinkedList<>();
        if(s.charAt(n-1)=='1') // can't reach destination
            return false;

        q.offer(0);
        int far = 0;
        while(!q.isEmpty()){
            int i = q.poll();
            int min = i + minJump;
            int max = i + maxJump;
            for(int j = Math.max(min,far); j <= max && j < n; j++) {
                if(s.charAt(j) == '0'){
                    q.offer(j);
                    if(j == n - 1)
                        return true; //reached at destination
                }
            }
            far = Math.max(far, max);
        }
        return false; 
    }

    // 377. Combination Sum IV

    // Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.

    // The test cases are generated so that the answer can fit in a 32-bit integer.

    // Example 1:

    // Input: nums = [1,2,3], target = 4
    // Output: 7
    // Explanation:
    // The possible combination ways are:
    // (1, 1, 1, 1)
    // (1, 1, 2)
    // (1, 2, 1)
    // (1, 3)
    // (2, 1, 1)
    // (2, 2)
    // (3, 1)
    // Note that different sequences are counted as different combinations.

    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        for (int i = 1; i <= target; i++) {
            for (int num : nums) {
                if (num <= i) {
                  dp[i] += dp[i-num];  
                }
            }
        }
        return dp[target];
    }

    // 141. Linked List Cycle

    // Given head, the head of a linked list, determine if the linked list has a cycle in it.

    // There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

    // Return true if there is a cycle in the linked list. Otherwise, return false.
    
    // Example 1:

    // Input: head = [3,2,0,-4], pos = 1
    // Output: true
    // Explanation: There is a cycle in the linked list, where the tail connects to the 1st node (0-indexed).

    public boolean hasCycle(ListNode head) {
        ListNode slow = head;
        ListNode fast = head;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;

            if (slow == fast)
                return true;
        }

        return false;
    }

    // 148. Sort List

    // Given the head of a linked list, return the list after sorting it in ascending order.

    // Example 1:

    // Input: head = [4,2,1,3]
    // Output: [1,2,3,4]

    // 1. Merge sort O(nlogn)

    public ListNode sortList(ListNode head) {
        if(head==null || head.next==null){
            return head;
        }
        ListNode slow=head;
        ListNode fast=head;
        ListNode prev=null;

        while(fast!=null && fast.next!=null){
            prev=slow;
            slow=slow.next;
            fast=fast.next.next;
        }
        prev.next=null;
        ListNode l1=sortList(head);
        ListNode l2=sortList(slow);
        //System.out.println(l1.val);
        return mergeSort(l1,l2);
    }
    public static ListNode mergeSort(ListNode l1,ListNode l2){
        ListNode fh=null;
        ListNode ft=null;

        while(l1!=null && l2!=null){
            if(fh==null && ft==null){
                if(l1.val>l2.val){
                    fh=l2;
                    ft=l2;
                    l2=l2.next;
                }
                else{
                    fh=l1;
                    ft=l1;
                    l1=l1.next;
                }
            }
            if(l1!=null && l2!=null){
                if(l1.val<l2.val){
                    ft.next=l1;
                    ft=ft.next;
                    l1=l1.next;
                }
                else{
                    ft.next=l2;
                    ft=ft.next;
                    l2=l2.next;
                }
            }
        }
        if(l1!=null){
            ft.next=l1;
        }
        if(l2!=null){
            ft.next=l2;
        }
        return fh;
    }

    // 1882. Process Tasks Using Servers

    // You are given two 0-indexed integer arrays servers and tasks of lengths n​​​​​​ and m​​​​​​ respectively. servers[i] is the weight of the i​​​​​​th​​​​ server, and tasks[j] is the time needed to process the j​​​​​​th​​​​ task in seconds.

    // Tasks are assigned to the servers using a task queue. Initially, all servers are free, and the queue is empty.

    // At second j, the jth task is inserted into the queue (starting with the 0th task being inserted at second 0). As long as there are free servers and the queue is not empty, the task in the front of the queue will be assigned to a free server with the smallest weight, and in case of a tie, it is assigned to a free server with the smallest index.

    // If there are no free servers and the queue is not empty, we wait until a server becomes free and immediately assign the next task. If multiple servers become free at the same time, then multiple tasks from the queue will be assigned in order of insertion following the weight and index priorities above.

    // A server that is assigned task j at second t will be free again at second t + tasks[j].

    // Build an array ans​​​​ of length m, where ans[j] is the index of the server the j​​​​​​th task will be assigned to.

    // Return the array ans​​​​.

    

    // Example 1:

    // Input: servers = [3,3,2], tasks = [1,2,3,2,1,2]
    // Output: [2,2,0,2,1,2]
    // Explanation: Events in chronological order go as follows:
    // - At second 0, task 0 is added and processed using server 2 until second 1.
    // - At second 1, server 2 becomes free. Task 1 is added and processed using server 2 until second 3.
    // - At second 2, task 2 is added and processed using server 0 until second 5.
    // - At second 3, server 2 becomes free. Task 3 is added and processed using server 2 until second 5.
    // - At second 4, task 4 is added and processed using server 1 until second 5.
    // - At second 5, all servers become free. Task 5 is added and processed using server 2 until second 7.

    public int[] assignTasks(int[] servers, int[] tasks) {
        // if the weight of the 2 server's is not same, we put the lowest weight in the front.
        // if the weight is same, we sort by the index, we pick the lowest index value server
        PriorityQueue<int[]> availableServers = new PriorityQueue<>((a,b) -> a[1] != b[1] ? Integer.compare(a[1], b[1]) : Integer.compare(a[0], b[0]));

        int n = servers.length;

        for(int i=0; i<n; i++) { 
            availableServers.add(new int[]{i, servers[i], 0});
        }

        // need to sort based on the time that the server is busy, if the times are equal sort the server based on the index
        PriorityQueue<int[]> runningServers = new PriorityQueue<>((a,b) -> a[2] != b[2] ? Integer.compare(a[2], b[2]) : a[1] != b[1] ? Integer.compare(a[1], b[1]) : Integer.compare(a[0], b[0]));
        
        int m = tasks.length;
        
        int[] result = new int[m];
        
        // O(mlogn)
        for(int i=0; i<m; i++) {
            int timeToCompleteTask = tasks[i];
            
            // if running servers queue is not empty, peek the time from the queue, if its less than the time required to complete the current task, poll that server, else that server is busy doing its task. we will have to look from the available servers queue.
            while(!runningServers.isEmpty() && runningServers.peek()[2] <= i) {
                availableServers.add(runningServers.poll());
            }
            
            if(!availableServers.isEmpty()) {
                int[] curServer = availableServers.poll();
                result[i] = curServer[0];
                curServer[2] = i + timeToCompleteTask;
                runningServers.add(curServer);
            } else {
                int[] curServer = runningServers.poll();
                result[i] = curServer[0];
                curServer[2] = curServer[2] + timeToCompleteTask;
                runningServers.add(curServer);
            }
        }
        return result;
    }

    // 1834. Single-Threaded CPU

    // You are given n​​​​​​ tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTimei, processingTimei] means that the i​​​​​​th​​​​ task will be available to process at enqueueTimei and will take processingTimei to finish processing.

    // You have a single-threaded CPU that can process at most one task at a time and will act in the following way:

    // If the CPU is idle and there are no available tasks to process, the CPU remains idle.
    // If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
    // Once a task is started, the CPU will process the entire task without stopping.
    // The CPU can finish a task then start a new one instantly.
    // Return the order in which the CPU will process the tasks.

    // Example 1:

    // Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
    // Output: [0,2,3,1]
    // Explanation: The events go as follows: 
    // - At time = 1, task 0 is available to process. Available tasks = {0}.
    // - Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
    // - At time = 2, task 1 is available to process. Available tasks = {1}.
    // - At time = 3, task 2 is available to process. Available tasks = {1, 2}.
    // - Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. Available tasks = {1}.
    // - At time = 4, task 3 is available to process. Available tasks = {1, 3}.
    // - At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. Available tasks = {1}.
    // - At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
    // - At time = 10, the CPU finishes task 1 and becomes idle.

    public int[] getOrder(int[][] tasks) {
        HashMap<int[],Integer> hm = new HashMap<int[],Integer>();
        for(int i=0;i<tasks.length;i++){
            hm.put(tasks[i],i); 
        } //HashMap for remembering the index the tasks as we want to sort
        Arrays.sort(tasks, (a, b) -> a[0] - b[0]); //Sorting the tasks based on enqueue time
        ArrayList<Integer> al = new ArrayList<Integer>(); //Using ArrayList just for ease
        PriorityQueue<int[]> heap = new PriorityQueue<>((o1, o2) -> { 
            int firstCompare = Integer.compare(o1[1], o2[1]);
            if(firstCompare == 0) {
                return Integer.compare(hm.get(o1), hm.get(o2));
            }
            return firstCompare;
        }); //PriorityQueue that will return the task with least processing time of those currently present in the queue and on 2 or more tasks having same processing time will return the one with lower index (using HashMap)
        int time=tasks[0][0]; //Start time will be the enqueue time of the 1st task from the sorted tasks matrix (i.e cpu processing starts from that time)
        for(int i=0;i<tasks.length;i++){
            int a = tasks[i][0]; //enqueue time of the element
            while(time<a&&!heap.isEmpty()){
                time+=heap.peek()[1];
                al.add(hm.get(heap.poll()));
            } //#3rd condition - we'll have to keep processing the task from the queue until time overlaps with the enqueue time of the upcoming tasks
            if(time<a&&heap.isEmpty()){
                time=tasks[i][0];
            } //4th condition - if there are no tasks to process in the queue (i.e. if the queue is empty) we'll have to assign the time as enqueue time of the current task like we did during intiation of time variable(basically no overlapping and stuffs so we starting from first again)
            for(int j=i;j<tasks.length;j++){
                if(tasks[j][0]==a){
                    heap.add(tasks[j]);
                    i=j;
                    continue;
                }
                break;
            } // #1st condition - Adding all the tasks to the queue with the same enqueue time
            if(a>=time){
                time+=heap.peek()[1];
                al.add(hm.get(heap.poll()));
            } //#2nd condition - if the enqueue time of the current tasks those we added into the queue is greater than or equal to the time it will process task from the queue(So this basically means when the cpu is available to process the tasks it'll process the one with lowest processing time present in the queue)
        }
        while(!heap.isEmpty()){
            al.add(hm.get(heap.poll()));
        } //This additional while loop is for completeing the pending tasks those are present in the queue
        int[] out = new int[al.size()];
        for(int i=0;i<al.size();i++){
            out[i]=al.get(i);
        } //Converting the ArrayList int Array 
        return out; //PEace
    }

    // 739. Daily Temperatures

    // Given an array of integers temperatures represents the daily temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature. If there is no future day for which this is possible, keep answer[i] == 0 instead.
    
    // Example 1:

    // Input: temperatures = [73,74,75,71,69,72,76,73]
    // Output: [1,1,4,2,1,1,0,0]
    // Example 2:

    // Input: temperatures = [30,40,50,60]
    // Output: [1,1,1,0]

    public int[] dailyTemperatures(int[] temperatures) {
        int n = temperatures.length;
        int res[] = new int[n];
        Stack<Integer> st = new Stack<>();
        for(int i = 0; i < n; i++) {
            while(!st.isEmpty() && temperatures[i] > temperatures[st.peek()]) {
                res[st.peek()] = i - st.peek();
                st.pop();
            }
            st.push(i);
        }
        return res;
    }

    // 256 Paint House

    // [[17,2,17],
    //  [16,16,5],
    //  [14,3,19]] -> output 10.
    // paint house 0 into blue
    // paint house 1 into green
    // paint house 2 into blue
    // find the min cost to paint house

    public int paintHouse(int[][] costs) {
        if (costs == null || costs.length == 0)
            return 0;
        for (int i = 1; i < costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        return Math.min(Math.min(costs[costs.length - 1][0], costs[costs.length - 1][1]), costs[costs.length - 1][2]);
    }


    // 1930. Unique Length-3 Palindromic Subsequences

    // Given a string s, return the number of unique palindromes of length three that are a subsequence of s.

    // Note that even if there are multiple ways to obtain the same subsequence, it is still only counted once.

    // A palindrome is a string that reads the same forwards and backwards.

    // A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

    // For example, "ace" is a subsequence of "abcde".
    

    // Example 1:

    // Input: s = "aabca"
    // Output: 3
    // Explanation: The 3 palindromic subsequences of length 3 are:
    // - "aba" (subsequence of "aabca")
    // - "aaa" (subsequence of "aabca")
    // - "aca" (subsequence of "aabca")

    public int countPalindromicSubsequence(String s) {
        int[] start = new int[26];
        int[] end = new int[26];
        Arrays.fill(start, -1);
        Arrays.fill(end, -1);
        for(int i = 0; i < s.length(); i++) {
            if(start[s.charAt(i) - 'a'] == -1) {
                start[s.charAt(i) - 'a'] = i;
            }
            end[s.charAt(i) - 'a'] = i;
        }

        int ans=0;
        for(int i = 0; i < 26; i++) {
            HashSet<Character> set = new HashSet<>();
            int st = start[i];
            int e = end[i];
            if(st != e && st != -1 && e != -1) {
                for(int j = st + 1; j < e; j++) {
                    if(!set.contains(s.charAt(j))) {
                        set.add(s.charAt(j));
                    }
                }    
            }
            ans += set.size();
            start[i] = -1;
            end[i] = -1;
        }
        return ans;
    }

    // 289. Game of Life

    // According to Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

    // The board is made up of an m x n grid of cells, where each cell has an initial state: live (represented by a 1) or dead (represented by a 0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

    // Any live cell with fewer than two live neighbors dies as if caused by under-population.
    // Any live cell with two or three live neighbors lives on to the next generation.
    // Any live cell with more than three live neighbors dies, as if by over-population.
    // Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
    // The next state is created by applying the above rules simultaneously to every cell in the current state, where births and deaths occur simultaneously. Given the current state of the m x n grid board, return the next state.

    // Example 1:

    // Input: board = [[0,1,0],[0,0,1],[1,1,1],[0,0,0]]
    // Output: [[0,0,0],[1,0,1],[0,1,1],[0,1,0]]



    private static final int[][] DIRS = {{-1, -1}, // top-left
                                        {-1, 0},  // top
                                        {-1, 1},  // top-right
                                        {0, -1},  // left
                                        {0, 1},   // right
                                        {1, -1},  // bottom-left
                                        {1, 0},   // bottom
                                        {1, 1}};  // bottom-right

    public void gameOfLife(int[][] board) {
        var rows = board.length;
        var cols = board[0].length;
        playGame(board, rows, cols);
        updateBoard(board, rows, cols);
    }

    private void playGame(int[][] board, int rows, int cols) {
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++) {
                var alive = aliveNeighbors(board, rows, cols, i, j);
                // Dead cell with 3 live neighbors becomes alive
                if (board[i][j] == 0 && alive == 3) // board[i][j] = 00
                    board[i][j] = 2; // board[i][j] = 10
                // Live cell with 2 or 3 live neighbors lives on
                else if (board[i][j] == 1 && (alive == 2 || alive == 3)) // board[i][j] = 01
                    board[i][j] = 3; // board[i][j] = 11
            }
    }

    private int aliveNeighbors(int[][] board, int rows, int cols, int i, int j) {
        var alive = 0;
        for (var dir : DIRS) {
            var neighborX = i + dir[0];
            var neighborY = j + dir[1];
            if (!isOutOfBounds(rows, cols, neighborX, neighborY))
                alive += board[neighborX][neighborY] & 1;
        }
        return alive;
    }

    private boolean isOutOfBounds(int rows, int cols, int x, int y) {
        return x < 0 || x >= rows || y < 0 || y >= cols;
    }

    private void updateBoard(int[][] board, int rows, int cols) {
        for (var i = 0; i < rows; i++)
            for (var j = 0; j < cols; j++)
                if (board[i][j] != 0) // this check is not necessary but improves efficiency
                    board[i][j] >>= 1; // right shift 1 bit to replace old state with new state
    }


    // 1849. Splitting a String Into Descending Consecutive Values

    // You are given a string s that consists of only digits.

    // Check if we can split s into two or more non-empty substrings such that the numerical values of the substrings are in descending order and the difference between numerical values of every two adjacent substrings is equal to 1.

    // For example, the string s = "0090089" can be split into ["0090", "089"] with numerical values [90,89]. The values are in descending order and adjacent values differ by 1, so this way is valid.
    // Another example, the string s = "001" can be split into ["0", "01"], ["00", "1"], or ["0", "0", "1"]. However all the ways are invalid because they have numerical values [0,1], [0,1], and [0,0,1] respectively, all of which are not in descending order.
    // Return true if it is possible to split s​​​​​​ as described above, or false otherwise.

    // A substring is a contiguous sequence of characters in a string.

    

    // Example 1:

    // Input: s = "1234"
    // Output: false
    // Explanation: There is no valid way to split s.
    // Example 2:

    // Input: s = "050043"
    // Output: true
    // Explanation: s can be split into ["05", "004", "3"] with numerical values [5,4,3].
    // The values are in descending order with adjacent values differing by 1.

    public boolean splitString(String s) {
        if (s == null || s.length() <= 1) return false;
        return backtrackSplit(0, s, new ArrayList<Long>());
    }

    public boolean backtrackSplit(int pos, String s, ArrayList<Long> list) {
        // Base case where we reach till end of string and we have atleast 2 parts
        if (pos >= s.length()) return list.size() >= 2;

        long num = 0;
        for (int i = pos; i < s.length(); i++) {
            num = num * 10 + (s.charAt(i) - '0'); // "070" i = 1 -> 0.. i = 2 -> 7.. i =3 -> 70 
            if (list.size()==0 || list.get(list.size()-1) - num == 1) { // if it is first digit or difference is +1 valid
                list.add(num);  // add the number and continue to next index
                if (backtrackSplit(i + 1, s, list)) 
                    return true;
                list.remove(list.size() - 1); // backtrack, done with that itteration coun't find it
            }
        }
         return false;
    }

    // 64. Minimum Path Sum

    // Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right, which minimizes the sum of all numbers along its path.

    // Note: You can only move either down or right at any point in time.

    // Example 1:

    // Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
    // Output: 7
    // Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
    // Example 2:

    // Input: grid = [[1,2,3],[4,5,6]]
    // Output: 12

    public int minPathSum(int[][] grid) {
        int m = grid.length;
        int n = grid[0].length;

        for (int i = 1; i < m; i++) {
            grid[i][0] += grid[i-1][0];
        }

        for (int j = 1; j < n; j++) {
            grid[0][j] += grid[0][j-1];
        }

        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                grid[i][j] += Math.min(grid[i-1][j], grid[i][j-1]);
            }
        }

        return grid[m-1][n-1];
    }

    // 1968. Array With Elements Not Equal to Average of Neighbors

    // You are given a 0-indexed array nums of distinct integers. You want to rearrange the elements in the array such that every element in the rearranged array is not equal to the average of its neighbors.

    // More formally, the rearranged array should have the property such that for every i in the range 1 <= i < nums.length - 1, (nums[i-1] + nums[i+1]) / 2 is not equal to nums[i].

    // Return any rearrangement of nums that meets the requirements.

    // Example 1:

    // Input: nums = [1,2,3,4,5]
    // Output: [1,2,4,5,3]
    // Explanation:
    // When i=1, nums[i] = 2, and the average of its neighbors is (1+4) / 2 = 2.5.
    // When i=2, nums[i] = 4, and the average of its neighbors is (2+5) / 2 = 3.5.
    // When i=3, nums[i] = 5, and the average of its neighbors is (4+3) / 2 = 3.5.

    public int[] rearrangeArray(int[] nums) {
        Arrays.sort(nums);
        // sort in wave format
        for(int i = 0;i<nums.length-1;i+=2){
            int temp = nums[i];
            nums[i] = nums[i+1];
            nums[i+1] = temp;
        }
        return nums;
    }

    // 75. Sort Colors

    // Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

    // We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

    // You must solve this problem without using the library's sort function.

    // Example 1:

    // Input: nums = [2,0,2,1,1,0]
    // Output: [0,0,1,1,2,2]
    // Example 2:

    // Input: nums = [2,0,1]
    // Output: [0,1,2]

    public void sortColors(int[] nums) {

        int low = 0;
        int mid = 0;
        int high = nums.length-1;

        while(mid <= high){
            if(nums[mid] == 0){
                swap(nums,mid,low);
                mid++;
                low++;
            }
            else if(nums[mid] == 2){
                swap(nums,mid,high);
                high--;
            }
            else{
                mid++;
            }
        }
    }

    private void swap(int[] arr, int i, int j){
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    // 268. Missing Number

    // Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.

    // Example 1:

    // Input: nums = [3,0,1]
    // Output: 2
    // Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
    // Example 2:

    // Input: nums = [0,1]
    // Output: 2
    // Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.

    public int missingNumber(int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; i++) {
            // XOR (2 ^ 3) (10 ^ 11) -> (01)
            // (5 ^ 5) (101 ^ 101) -> (000) 
            // (5 ^ 5 ^ 3) -> (3)
            // ([0,1,2,3] ^[0,1,3] -> [2])
            missing = missing ^ (i ^ nums[i]);
        }
        return missing;
    }


    // 271. Encode and Decode Strings

    // Encodes a list of strings to a single string.

    public String encode(List<String> strs) {
        if (strs.size() == 0)
            return Character.toString((char) 258); // Ā

        String d = Character.toString((char) 257); // ā
        StringBuilder sb = new StringBuilder();
        for (String s : strs) {
            sb.append(s);
            sb.append(d);
        }
        sb.deleteCharAt(sb.length() - 1);
        return sb.toString();
    }

    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        String d = Character.toString((char) 258);
        if (s.equals(d))
            return new ArrayList();

        d = Character.toString((char) 257);
        return Arrays.asList(s.split(d, -1));
    }

    // 1898. Maximum Number of Removable Characters

    // You are given two strings s and p where p is a subsequence of s. You are also given a distinct 0-indexed integer array removable containing a subset of indices of s (s is also 0-indexed).

    // You want to choose an integer k (0 <= k <= removable.length) such that, after removing k characters from s using the first k indices in removable, p is still a subsequence of s. More formally, you will mark the character at s[removable[i]] for each 0 <= i < k, then remove all marked characters and check if p is still a subsequence.

    // Return the maximum k you can choose such that p is still a subsequence of s after the removals.

    // A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

    

    // Example 1:

    // Input: s = "abcacb", p = "ab", removable = [3,1,0]
    // Output: 2
    // Explanation: After removing the characters at indices 3 and 1, "abcacb" becomes "accb".
    // "ab" is a subsequence of "accb".
    // If we remove the characters at indices 3, 1, and 0, "abcacb" becomes "ccb", and "ab" is no longer a subsequence.
    // Hence, the maximum k is 2.


    public int maximumRemovals(String s, String p, int[] removable) {
        int l = -1;
        int r = removable.length;
        char[] pArray = p.toCharArray();
        char[] sArray = s.toCharArray();
        int start = -1;
        while(l + 1 < r) {
            int m = l + ((r - l) >> 1);
            if (isSub(sArray, pArray, removable, start + 1, m)) {
                l = m;
                start = l;
            } else {
                r = m;
                sArray = s.toCharArray();
                start = -1;
            }
        }
        return l + 1;
    }
    public boolean isSub(char[] s, char[] p, int[] removable, int start, int k) {
        for (int i = start; i <= k; i++) {
            s[removable[i]] = '.';
        }
        int i = 0, j = 0;
        while (i < s.length && j < p.length) {
            if (s[i] == p[j]) {
                i++;
                j++;
            } else {
                i++;
            }
        }
        return j == p.length;
    }

    // 1423. Maximum Points You Can Obtain from Cards

    // There are several cards arranged in a row, and each card has an associated number of points. The points are given in the integer array cardPoints.

    // In one step, you can take one card from the beginning or from the end of the row. You have to take exactly k cards.

    // Your score is the sum of the points of the cards you have taken.

    // Given the integer array cardPoints and the integer k, return the maximum score you can obtain.

    

    // Example 1:

    // Input: cardPoints = [1,2,3,4,5,6,1], k = 3
    // Output: 12
    // Explanation: After the first step, your score will always be 1. However, choosing the rightmost card first will maximize your total score. The optimal strategy is to take the three cards on the right, giving a final score of 1 + 6 + 5 = 12.

    public int maxScore(int[] cardPoints, int k) {
        int start = 0;
        int end = 0;
        int sum = 0;
        k = cardPoints.length - k;
        int ans = Integer.MAX_VALUE;
        int totalSum = 0;

        for(int i = 0; i < cardPoints.length; i++) {
            totalSum += cardPoints[i];
        }

        if(k == 0) {
            return totalSum;
        }

        while(end < cardPoints.length) {

            sum += cardPoints[end];
            if(end - start + 1 < k) {
                end++;
            }
            else if(end - start + 1 == k) {
                ans = Math.min(ans, sum);
                sum -= cardPoints[start];
                start++;
                end++;
            }
        }
        return totalSum - ans;
    }










    // 100 problems so far















    // 518. Coin Change II

    // You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

    // Return the number of combinations that make up that amount. If that amount of money cannot be made up by any combination of the coins, return 0.

    // You may assume that you have an infinite number of each kind of coin.

    // The answer is guaranteed to fit into a signed 32-bit integer.

    // Example 1:

    // Input: amount = 5, coins = [1,2,5]
    // Output: 4
    // Explanation: there are four ways to make up the amount:
    // 5=5
    // 5=2+2+1
    // 5=2+1+1+1
    // 5=1+1+1+1+1

    public int change(int amount, int[] coins) {
        int[] dp = new int[amount + 1];
        dp[0] = 1;
        
        for (int coin : coins) {
            for (int j = coin; j <= amount; j++) {
                dp[j] += dp[j - coin];
            }
        }
        
        return dp[amount];
    }

    // 1905. Count Sub Islands

    // You are given two m x n binary matrices grid1 and grid2 containing only 0's (representing water) and 1's (representing land). An island is a group of 1's connected 4-directionally (horizontal or vertical). Any cells outside of the grid are considered water cells.

    // An island in grid2 is considered a sub-island if there is an island in grid1 that contains all the cells that make up this island in grid2.

    // Return the number of islands in grid2 that are considered sub-islands.

    // Example 1:

    // Input: grid1 = [[1,1,1,0,0],[0,1,1,1,1],[0,0,0,0,0],[1,0,0,0,0],[1,1,0,1,1]], grid2 = [[1,1,1,0,0],[0,0,1,1,1],[0,1,0,0,0],[1,0,1,1,0],[0,1,0,1,0]]
    // Output: 3
    // Explanation: In the picture above, the grid on the left is grid1 and the grid on the right is grid2.
    // The 1s colored red in grid2 are those considered to be part of a sub-island. There are three sub-islands.






    // 61. Rotate List

    // Given the head of a linked list, rotate the list to the right by k places.

    // Example 1:

    // Input: head = [1,2,3,4,5], k = 2
    // Output: [4,5,1,2,3]

    public ListNode rotateRight(ListNode head, int k) {
        if(head == null)
            return null;

        int len = getLength(head);
        k = k % len;
        ListNode fast = head, slow = head;
        while(k != 0){
           fast = fast.next;
            k--;
        }
        while(fast.next != null){
            slow = slow.next;
            fast = fast.next;
        }
        
        fast.next=head;
        head=slow.next;
        slow.next=null;
        return head;
    }

    public int getLength(ListNode head){
        int len = 0;
        while(head != null){
            len++;
            head = head.next;
        }
        return len;
    }

    // 50. Pow(x, n)

    Implement pow(x, n), which calculates x raised to the power n (i.e., xn).

    Example 1:

    Input: x = 2.00000, n = 10
    Output: 1024.00000
    Example 2:

    Input: x = 2.10000, n = 3
    Output: 9.26100
    Example 3:

    Input: x = 2.00000, n = -2
    Output: 0.25000
    Explanation: 2^-2 = 1/2^2 = 1/4 = 0.25


    public double myPow(double x, int n) {
        long longN = n; // Convert n to a long to handle Integer.MIN_VALUE
        double ans = solve(x, Math.abs(longN));

        if (longN < 0) {
            return 1 / ans;
        }
        return ans;
    }

    public double solve(double x, long n) {
        if (n == 0) {
            return 1; // power of 0 is 1
        }
        double temp = solve(x, n / 2);
        temp = temp * temp;
        if (n % 2 == 0) { // if even, return just without doing anything
            return temp;
        } else {
            return temp * x; // if odd, return by multiplying once more with the given number
        }
    }

    // 973. K Closest Points to Origin

    // Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

    // The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

    // You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).

    // Input: points = [[1,3],[-2,2]], k = 1
    // Output: [[-2,2]]
    // Explanation:
    // The distance between (1, 3) and the origin is sqrt(10).
    // The distance between (-2, 2) and the origin is sqrt(8).
    // Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
    // We only want the closest k = 1 points from the origin, so the answer is just [[-2,2]].



    public int[][] kClosest(int[][] points, int k) {
        PriorityQueue<int[]> pq = new PriorityQueue<>((a,b)->a[1]-b[1]);

        for(int i = 0; i < points.length; i++){
            int distance = points[i][0] * points[i][0] + points[i][1] * points[i][1];
            pq.add(new int[] {i, distance});
        }

        int[][] res = new int[k][];
        while(k > 0){
            res[k-1] = points[pq.poll()[0]];
            k--;
        }
        return res;
    }

    // 1921. Eliminate Maximum Number of Monsters

    // You are playing a video game where you are defending your city from a group of n monsters. You are given a 0-indexed integer array dist of size n, where dist[i] is the initial distance in kilometers of the ith monster from the city.

    // The monsters walk toward the city at a constant speed. The speed of each monster is given to you in an integer array speed of size n, where speed[i] is the speed of the ith monster in kilometers per minute.

    // You have a weapon that, once fully charged, can eliminate a single monster. However, the weapon takes one minute to charge.The weapon is fully charged at the very start.

    // You lose when any monster reaches your city. If a monster reaches the city at the exact moment the weapon is fully charged, it counts as a loss, and the game ends before you can use your weapon.

    // Return the maximum number of monsters that you can eliminate before you lose, or n if you can eliminate all the monsters before they reach the city.

    

    // Example 1:

    // Input: dist = [1,3,4], speed = [1,1,1]
    // Output: 3
    // Explanation:
    // In the beginning, the distances of the monsters are [1,3,4]. You eliminate the first monster.
    // After a minute, the distances of the monsters are [X,2,3]. You eliminate the second monster.
    // After a minute, the distances of the monsters are [X,X,2]. You eliminate the thrid monster.
    // All 3 monsters can be eliminated.
    // Example 2:

    // Input: dist = [1,1,2,3], speed = [1,1,1,1]
    // Output: 1
    // Explanation:
    // In the beginning, the distances of the monsters are [1,1,2,3]. You eliminate the first monster.
    // After a minute, the distances of the monsters are [X,0,1,2], so you lose.
    // You can only eliminate 1 monster.


    public int eliminateMaximum(int[] dist, int[] speed) {
        int[] time = new int[dist.length];

        for(int i=0;i<dist.length;i++) {
            time[i] = (int)Math.ceil((double)dist[i] / speed[i]);
        }

        Arrays.sort(time);
        int ans = 0;
        int T = 0;
        for(int i = 0;i < time.length; i++) {
            if (T++ < time[i])
                ans++;
            else
                break;
        }
        return ans;
    }

    // 199. Binary Tree Right Side View

    // Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.

    // Example 1:

    // Input: root = [1,2,3,null,5,null,4]
    // Output: [1,3,4]

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

    // 134. Gas Station

    // There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].

    // You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.

    // Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique

    // Example 1:

    // Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
    // Output: 3
    // Explanation:
    // Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
    // Travel to station 4. Your tank = 4 - 1 + 5 = 8
    // Travel to station 0. Your tank = 8 - 2 + 1 = 7
    // Travel to station 1. Your tank = 7 - 3 + 2 = 6
    // Travel to station 2. Your tank = 6 - 4 + 3 = 5
    // Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
    // Therefore, return 3 as the starting index.

    public int canCompleteCircuit(int[] gas, int[] cost) {
        int amt = 0;
        int t = 0;
        int idx = 0;
        for (int i = 0; i < gas.length; i++){
            amt += gas[i] - cost[i];
            t += gas[i] - cost[i];
            if (amt < 0){
                amt = 0;
                idx = i + 1;
            }
        }
        if (t < 0){
            //incomplete path
            return -1;
        }
        return idx;
    }

    // 1980. Find Unique Binary String

    // Given an array of strings nums containing n unique binary strings each of length n, return a binary string of length n that does not appear in nums. If there are multiple answers, you may return any of them.
    
    // Example 1:

    // Input: nums = ["01","10"]
    // Output: "11"
    // Explanation: "11" does not appear in nums. "00" would also be correct.


    public static int n;
    public String findDifferentBinaryString(String[] nums) {
        HashSet<String> hs = new HashSet<>();
        ArrayList<String> ds = new ArrayList<>();
        n = nums[0].length();

        for(String str: nums)
            hs.add(str);

        for(String str: nums)
            diffHelper(0, hs, ds, new StringBuilder(str));

        return ds.get(0);
    }

    public static void diffHelper(int ind, HashSet<String> hs, ArrayList<String> ds, StringBuilder sb){
        if(ind == n){
            if(!hs.contains(sb.toString())) 
                ds.add(sb.toString());
            return;
        }
        if(ds.size() > 0) 
            return;
        for(int i = ind; i < n; i++) {
            //either flip the character at that index 
            sb.setCharAt(i,sb.charAt(i) == '0'?'1':'0');
            diffHelper(i+1, hs, ds, sb);
            sb.setCharAt(i,sb.charAt(i) == '0'?'1':'0');
            //or skip the character at that index
            diffHelper(i+1, hs, ds, sb);
        }
    }

    // 1985. Find the Kth Largest Integer in the Array

    // You are given an array of strings nums and an integer k. Each string in nums represents an integer without leading zeros.

    // Return the string that represents the kth largest integer in nums.

    // Note: Duplicate numbers should be counted distinctly. For example, if nums is ["1","2","2"], "2" is the first largest integer, "2" is the second-largest integer, and "1" is the third-largest integer.

    // Example 1:

    // Input: nums = ["3","6","7","10"], k = 4
    // Output: "3"
    // Explanation:
    // The numbers in nums sorted in non-decreasing order are ["3","6","7","10"].
    // The 4th largest integer in nums is "3".

    public String kthLargestNumber(String[] nums, int k) {
        //Making the priority queue using custom comparator(*Discussed below*)
        PriorityQueue<String> pq=new PriorityQueue<>((a,b) -> a.length() == b.length() ? a.compareTo(b) : a.length() - b.length());
        
        for(String n: nums){
            pq.add(n);
            //Deleting the kth largest everytime, so that pq contains only the largest k elements
            if(pq.size() > k) 
                pq.poll();
        }
        //Returning the minimum element in pq
        return pq.poll();
    }

    // 787. Cheapest Flights Within K 
    
    // There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

    // You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

    // Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
    // Output: 700
    // Explanation:
    // The graph is shown above.
    // The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
    // Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.

    // bellman ford algorithm

    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        List<List<int[]>> adj = new ArrayList<>();
        for (int i = 0; i < n; i++)  {
            adj.add(new ArrayList<>());
        }
        for (int[] flight : flights) {
            adj.get(flight[0]).add(new int[] {flight[1], flight[2]});
        }

        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[] {src, 0});

        int[] minCost = new int[n];
        Arrays.fill(minCost, Integer.MAX_VALUE);
        int stops = 0;

        while (!q.isEmpty() && stops <= k) {
            int size = q.size();
            while (size-- > 0) {
                int[] curr = q.poll();
                for (int[] neighbour : adj.get(curr[0])) {
                    int price = neighbour[1], neighbourNode = neighbour[0];
                    if (price + curr[1] >= minCost[neighbourNode])
                        continue;
                    minCost[neighbourNode] = price + curr[1];
                    q.offer(new int[] {neighbourNode, minCost[neighbourNode]});
                }
            }
            stops++;
        }
        return minCost[dst] == Integer.MAX_VALUE ? -1 : minCost[dst];
    }

    // 130. Surrounded Regions

    // Given an m x n matrix board containing 'X' and 'O', capture all regions that are 4-directionally surrounded by 'X'.

    // A region is captured by flipping all 'O's into 'X's in that surrounded region.

    // Input: board = [["X","X","X","X"],["X","O","O","X"],["X","X","O","X"],["X","O","X","X"]]
    // Output: [["X","X","X","X"],["X","X","X","X"],["X","X","X","X"],["X","O","X","X"]]
    // Explanation: Notice that an 'O' should not be flipped if:
    // - It is on the border, or
    // - It is adjacent to an 'O' that should not be flipped.
    // The bottom 'O' is on the border, so it is not flipped.
    // The other three 'O' form a surrounded region, so they are flipped.

    public void solve(char[][] board) {
        int ROWS = board.length;
        int COLS = board[0].length;

        // capture the unsurrounded regions and change them into "T" (O -> T)
        for(int r = 0; r < ROWS - 1; r++) {
            for(int c = 0; c < COLS - 1; c++) {
                if(board[r][c] == 'O' && ((r > 0 && r < ROWS - 1) || (c > 0 && c < COLS - 1)) ) {
                    capture(board, r, c, ROWS, COLS);
                }
            }
        }

        // Capture the surrounded regions (O -> X);
        for(int r = 0; r < ROWS - 1; r++) {
            for(int c = 0; c < COLS - 1; c++) {
                if(board[r][c] == 'O') {
                    board[r][c] = 'X';
                }
            }
        }

        // Uncapture unsorrounded regions (T -> O)
        for(int r = 0; r < ROWS - 1; r++) {
            for(int c = 0; c < COLS - 1; c++) {
                if(board[r][c] == 'T') {
                    board[r][c] = 'O';
                }
            }
        }
    }

    private void capture(char[][] board, int r, int c, int ROWS, int COLS) {
        if(r < 0 || c < 0 || r == ROWS || c == COLS || board[r][c] != 'O'){
            return;
        }
        board[r][c] = 'T';
        capture(board, r + 1, c, ROWS, COLS);
        capture(board, r - 1, c, ROWS, COLS);
        capture(board, r, c + 1, ROWS, COLS);
        capture(board, r, c - 1, ROWS, COLS);
    }



    // 1584. Min Cost to Connect All Points

    // You are given an array points representing integer coordinates of some points on a 2D-plane, where points[i] = [xi, yi].

    // The cost of connecting two points [xi, yi] and [xj, yj] is the manhattan distance between them: |xi - xj| + |yi - yj|, where |val| denotes the absolute value of val.

    // Return the minimum cost to make all points connected. All points are connected if there is exactly one simple path between any two points.

    

    // Example 1:


    // Input: points = [[0,0],[2,2],[3,10],[5,2],[7,0]]
    // Output: 20
    // Explanation: 

    // We can connect the points as shown above to get the minimum cost of 20.
    // Notice that there is a unique path between every pair of points.
    // Example 2:

    // Input: points = [[3,12],[-2,5],[-4,1]]
    // Output: 18

    // Prim's Algorithm
    // Min spanning tree

    Set<Integer> vis;
    PriorityQueue<Pair<int[], Integer>> pq;
    public void getDistances(int[][] points, int index){
        for(int i=0; i<points.length;i++){
            if(i!= index && (vis.contains(i) == false) ){
                int[] s= new int[]{index, i};
                int dist = Math.abs(points[i][0]-points[index][0]) + Math.abs(points[i][1]-points[index][1]);
                Pair<int[], Integer> p = new Pair<>(s, dist);
                pq.add(p);
            }
        }
    }
    public int minCostConnectPoints(int[][] points) {
        vis = new HashSet<>();
        pq = new PriorityQueue<>((a,b)->a.getValue()-b.getValue());
        vis.add(0);
        getDistances(points,0);
        int count =0;
        int cost = 0;
        
        while(count != points.length-1){
            Pair<int[], Integer> p = pq.poll();
            int[] s = p.getKey();
            int val = p.getValue();
            int point = s[1];
            if(vis.contains(point))
                continue;
            count+=1;
            cost+=val;
            vis.add(point);
            getDistances(points, point);
            
        }
        return cost;
    }

    // 2002. Maximum Product of the Length of Two Palindromic Subsequences

    // Given a string s, find two disjoint palindromic subsequences of s such that the product of their lengths is maximized. The two subsequences are disjoint if they do not both pick a character at the same index.

    // Return the maximum possible product of the lengths of the two palindromic subsequences.

    // A subsequence is a string that can be derived from another string by deleting some or no characters without changing the order of the remaining characters. A string is palindromic if it reads the same forward and backward.

    

    // Example 1:

    // example-1
    // Input: s = "leetcodecom"
    // Output: 9
    // Explanation: An optimal solution is to choose "ete" for the 1st subsequence and "cdc" for the 2nd subsequence.
    // The product of their lengths is: 3 * 3 = 9.

    public int maxProduct(String s) {
        char[] strArr = s.toCharArray();
        int n = strArr.length;
        Map<Integer, Integer> pali = new HashMap<>();
        // save all elligible combination into hashmap
        for (int mask = 0; mask < 1 << n; mask++){
            String subseq = "";
            for (int i = 0; i < 12; i++){
                if ((mask & 1<<i) > 0)
                    subseq += strArr[i];
            }
            if (isPalindromic(subseq))
                pali.put(mask, subseq.length());
        }
        // use & opertion between any two combination
        int res = 0;
        for (int mask1 : pali.keySet()){
            for (int mask2 : pali.keySet()){
                if ((mask1 & mask2) == 0)
                    res = Math.max(res, pali.get(mask1)*pali.get(mask2));
            }
        }

        return res;
    }

    public boolean isPalindromic(String str){
        int j = str.length() - 1;
        char[] strArr = str.toCharArray();
        for (int i = 0; i < j; i ++){
            if (strArr[i] != strArr[j])
                return false;
            j--;
        }
        return true;
    }

    // 77. Combinations

    // Given two integers n and k, return all possible combinations of k numbers chosen from the range [1, n].

    // You may return the answer in any order.

    // Example 1:

    // Input: n = 4, k = 2
    // Output: [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
    // Explanation: There are 4 choose 2 = 6 total combinations.
    // Note that combinations are unordered, i.e., [1,2] and [2,1] are considered to be the same combination.

    // O(k * n ^ k)

    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> combine(int n, int k) {
        List<Integer> ans = new ArrayList<>();
        comboSolve(1, n, k, ans);
        return res;
    }

    private void comboSolve(int num, int tot, int k, List<Integer> ans) {
        if (ans.size() == k) {
            res.add(new ArrayList<>(ans));
            return;
        }
        for (int i = num; i <= tot; i++) {
            ans.add(i);
            comboSolve(i + 1, tot, k, ans);
            ans.remove(ans.size() - 1);
        }
    }

    // 1984. Minimum Difference Between Highest and Lowest of K Scores

    // You are given a 0-indexed integer array nums, where nums[i] represents the score of the ith student. You are also given an integer k.

    // Pick the scores of any k students from the array so that the difference between the highest and the lowest of the k scores is minimized.

    // Return the minimum possible difference.

    

    // Example 1:

    // Input: nums = [90], k = 1
    // Output: 0
    // Explanation: There is one way to pick score(s) of one student:
    // - [90]. The difference between the highest and lowest score is 90 - 90 = 0.
    // The minimum possible difference is 0.

    public int minimumDifference(int[] nums, int k) {
        if (k == 1) return 0;

        Arrays.sort(nums);

        int i = 0, j = k - 1,
        min = Integer.MAX_VALUE;

        while (j < nums.length) {
            min = Math.min(min, nums[j++] - nums[i++]);
        }
        return min;
    }

    // 2001. Number of Pairs of Interchangeable Rectangles

    // You are given n rectangles represented by a 0-indexed 2D integer array rectangles, where rectangles[i] = [widthi, heighti] denotes the width and height of the ith rectangle.

    // Two rectangles i and j (i < j) are considered interchangeable if they have the same width-to-height ratio. More formally, two rectangles are interchangeable if widthi/heighti == widthj/heightj (using decimal division, not integer division).

    // Return the number of pairs of interchangeable rectangles in rectangles.

    

    // Example 1:

    // Input: rectangles = [[4,8],[3,6],[10,20],[15,30]]
    // Output: 6
    // Explanation: The following are the interchangeable pairs of rectangles by index (0-indexed):
    // - Rectangle 0 with rectangle 1: 4/8 == 3/6.
    // - Rectangle 0 with rectangle 2: 4/8 == 10/20.
    // - Rectangle 0 with rectangle 3: 4/8 == 15/30.
    // - Rectangle 1 with rectangle 2: 3/6 == 10/20.
    // - Rectangle 1 with rectangle 3: 3/6 == 15/30.
    // - Rectangle 2 with rectangle 3: 10/20 == 15/30.


    // O(k * n ^ k)

    public long interchangeableRectangles(int[][] rectangles) {
        // return countRectangles(getRatios(rectangles));
        int n = rectangles.length;
        Map<Double, Long> count = new HashMap<>();
        for(int i=0;i<n;i++){
            double ratio = rectangles[i][0] / (rectangles[i][1] / 1.0);
            long y = 0;
            count.put(ratio,count.getOrDefault(ratio, y) + 1);
        }
        long points = 0;
        for(long k : count.values()){
            if (k > 1) {
                points += k * (k - 1) / 2;
            }
        }
        return points;
    }

    // 535. Encode and Decode TinyURL

    // Note: This is a companion problem to the System Design problem: Design TinyURL.
    // TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl and it returns a short URL such as http://tinyurl.com/4e9iAk. Design a class to encode a URL and decode a tiny URL.

    // There is no restriction on how your encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL and the tiny URL can be decoded to the original URL.

    // Implement the Solution class:

    // Solution() Initializes the object of the system.
    // String encode(String longUrl) Returns a tiny URL for the given longUrl.
    // String decode(String shortUrl) Returns the original long URL for the given shortUrl. It is guaranteed that the given shortUrl was encoded by the same object.
    

    // Example 1:

    // Input: url = "https://leetcode.com/problems/design-tinyurl"
    // Output: "https://leetcode.com/problems/design-tinyurl"

    // Explanation:
    // Solution obj = new Solution();
    // string tiny = obj.encode(url); // returns the encoded tiny url.
    // string ans = obj.decode(tiny); // returns the original url after decoding it.


    public class Codec {

        // Map containing URLs 
        private HashMap<String, String> urlMap = new HashMap<>(); 
        
        // Domain to return the shortened URL 
        private static final String DOMAIN = "https://tinyurl.com/";
        
        //  Length of random string 
        private static final int LENGTH = 5; 

        // Encodes a URL to a shortened URL.
        public String encode(String longUrl) {
            String code = null; 
            
            // Making sure that "randomly" generated key is not in the map
            do  {
                code = getRandomString(LENGTH);    
            } while(urlMap.containsKey(code));
            
            // Adding new entry
            urlMap.put(code, longUrl);

            // Returning the shortened URL
            return DOMAIN + code;
        }

        // Decodes a shortened URL to its original URL.
        public String decode(String shortUrl) {
            // Returning the URL stored in the map.
            return urlMap.get(
                shortUrl.substring(shortUrl.lastIndexOf('/')+1) 
            );
        }

        // Getting random String
        private String getRandomString(int length) {
            StringBuilder builder = new StringBuilder();
            while(length-- > 0) {
                builder.append((int) (Math.random() * 9 + 1));
            }
            return builder.toString();
        }
    }


    // 1899. Merge Triplets to Form Target Triplet

    // A triplet is an array of three integers. You are given a 2D integer array triplets, where triplets[i] = [ai, bi, ci] describes the ith triplet. You are also given an integer array target = [x, y, z] that describes the triplet you want to obtain.

    // To obtain target, you may apply the following operation on triplets any number of times (possibly zero):

    // Choose two indices (0-indexed) i and j (i != j) and update triplets[j] to become [max(ai, aj), max(bi, bj), max(ci, cj)].
    // For example, if triplets[i] = [2, 5, 3] and triplets[j] = [1, 7, 5], triplets[j] will be updated to [max(2, 1), max(5, 7), max(3, 5)] = [2, 7, 5].
    // Return true if it is possible to obtain the target triplet [x, y, z] as an element of triplets, or false otherwise.

    

    // Example 1:

    // Input: triplets = [[2,5,3],[1,8,4],[1,7,5]], target = [2,7,5]
    // Output: true
    // Explanation: Perform the following operations:
    // - Choose the first and last triplets [[2,5,3],[1,8,4],[1,7,5]]. Update the last triplet to be [max(2,1), max(5,7), max(3,5)] = [2,7,5]. triplets = [[2,5,3],[1,8,4],[2,7,5]]
    // The target triplet [2,7,5] is now an element of triplets.

    public boolean mergeTriplets(int[][] triplets, int[] target) {
        // A set to store the matching indexes
        HashSet<Integer> set=new HashSet<>();
        for(int triplet[]: triplets) {
            // if the any of the triplet triplet[0,1,2] has a greater value than the target[0,1,2] it means this cant be merged as it has a greater value (max will always return a greater value and not the value in the target) soo we reject all those triplets
            if(triplet[0]>target[0] || triplet[1]>target[1] || triplet[2]>target[2]) {
                continue;
            }
            // We need to find if the other values exist in the remaining positions coz if we remove all the triplets having values greater than target than when we merge the remaining triplets we have to get the target if the elements of the target is present
            for(int i = 0;i < 3; i++) {
                if(triplet[i]==target[i]) {
                    set.add(i);
                }
            }
        }
        return set.size() == 3;
    }

    // 114. Flatten Binary Tree to Linked List

    // Given the root of a binary tree, flatten the tree into a "linked list":

    // The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
    // The "linked list" should be in the same order as a pre-order traversal of the binary tree.
    

    // Example 1:


    // Input: root = [1,2,5,3,4,null,6]
    // Output: [1,null,2,null,3,null,4,null,5,null,6]

    public void flatten(TreeNode root) {
        flattenHelper(root, null);
    }

    public TreeNode flattenHelper(TreeNode root, TreeNode prev) {
        // base case
        if (root == null)
            return prev;
        
        // hypothesis step
        TreeNode right = flattenHelper(root.right, prev);
		// for left subtree prev would be the node that we get from the right subtree recursion
        TreeNode left = flattenHelper(root.left, right);
        
        // induction step
        root.right = left;
        root.left = null;
        
        return root;
    }

    // 1845. Seat Reservation Manager

    // Design a system that manages the reservation state of n seats that are numbered from 1 to n.

    // Implement the SeatManager class:

    // SeatManager(int n) Initializes a SeatManager object that will manage n seats numbered from 1 to n. All seats are initially available.
    // int reserve() Fetches the smallest-numbered unreserved seat, reserves it, and returns its number.
    // void unreserve(int seatNumber) Unreserves the seat with the given seatNumber.
    

    // Example 1:

    // Input
    // ["SeatManager", "reserve", "reserve", "unreserve", "reserve", "reserve", "reserve", "reserve", "unreserve"]
    // [[5], [], [], [2], [], [], [], [], [5]]
    // Output
    // [null, 1, 2, null, 2, 3, 4, 5, null]

    // Explanation
    // SeatManager seatManager = new SeatManager(5); // Initializes a SeatManager with 5 seats.
    // seatManager.reserve();    // All seats are available, so return the lowest numbered seat, which is 1.
    // seatManager.reserve();    // The available seats are [2,3,4,5], so return the lowest of them, which is 2.
    // seatManager.unreserve(2); // Unreserve seat 2, so now the available seats are [2,3,4,5].
    // seatManager.reserve();    // The available seats are [2,3,4,5], so return the lowest of them, which is 2.
    // seatManager.reserve();    // The available seats are [3,4,5], so return the lowest of them, which is 3.
    // seatManager.reserve();    // The available seats are [4,5], so return the lowest of them, which is 4.
    // seatManager.reserve();    // The only available seat is seat 5, so return 5.
    // seatManager.unreserve(5); // Unreserve seat 5, so now the available seats are [5].

    class SeatManager {

        PriorityQueue<Integer> pq;
        int count;

        public SeatManager(int n) {
            count = 1;
            pq = new PriorityQueue();
        }
        
        public int reserve() {
            if(pq.size()==0)
                return count++;
            return pq.poll();
        }
        
        public void unreserve(int seatNumber) {
            pq.add(seatNumber);
        }
    }

    // 2013. Detect Squares

    // You are given a stream of points on the X-Y plane. Design an algorithm that:

    // Adds new points from the stream into a data structure. Duplicate points are allowed and should be treated as different points.
    // Given a query point, counts the number of ways to choose three points from the data structure such that the three points and the query point form an axis-aligned square with positive area.
    // An axis-aligned square is a square whose edges are all the same length and are either parallel or perpendicular to the x-axis and y-axis.

    // Implement the DetectSquares class:

    // DetectSquares() Initializes the object with an empty data structure.
    // void add(int[] point) Adds a new point point = [x, y] to the data structure.
    // int count(int[] point) Counts the number of ways to form axis-aligned squares with point point = [x, y] as described above.
    

    // Example 1:


    // Input
    // ["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
    // [[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
    // Output
    // [null, null, null, null, 1, 0, null, 2]

    class DetectSquares {

        List<int[]> coordinates;
        Map<String, Integer> counts;

        public DetectSquares() {
            coordinates = new ArrayList<>();
            counts = new HashMap<>();
        }
        
        public void add(int[] point) {
            coordinates.add(point);
            String key = point[0] + "@" + point[1];
            counts.put(key, counts.getOrDefault(key, 0) + 1);
        }
        
        public int count(int[] point) {
            int sum = 0, px = point[0], py = point[1];
            for (int[] coordinate : coordinates) {
                int x = coordinate[0], y = coordinate[1];
                if (px == x || py == y || (Math.abs(px - x) != Math.abs(py - y)))
                    continue;
                sum += counts.getOrDefault(x + "@" + py, 0) * counts.getOrDefault(px + "@" + y, 0);
            }
            return sum;
        }
    }

    // 494. Target Sum

    // You are given an integer array nums and an integer target.

    // You want to build an expression out of nums by adding one of the symbols '+' and '-' before each integer in nums and then concatenate all the integers.

    // For example, if nums = [2, 1], you can add a '+' before 2 and a '-' before 1 and concatenate them to build the expression "+2-1".
    // Return the number of different expressions that you can build, which evaluates to target.

    

    // Example 1:

    // Input: nums = [1,1,1,1,1], target = 3
    // Output: 5
    // Explanation: There are 5 ways to assign symbols to make the sum of nums be target 3.
    // -1 + 1 + 1 + 1 + 1 = 3
    // +1 - 1 + 1 + 1 + 1 = 3
    // +1 + 1 - 1 + 1 + 1 = 3
    // +1 + 1 + 1 - 1 + 1 = 3
    // +1 + 1 + 1 + 1 - 1 = 3


    public int findTargetSumWays(int[] nums, int target) {
        return sumHelper(nums,0,target,0);
    }
    private int sumHelper(int[] nums,int ans,int target,int index){
        if(index == nums.length && ans != target) return 0;
        if(index == nums.length && ans == target){
            return 1;
        }
       int left = sumHelper(nums,ans + nums[index],target,index+1);
       int right = sumHelper(nums,ans - nums[index],target,index+1);
       return left+right;
    }



}