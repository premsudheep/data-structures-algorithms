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

public class DynamicProgramming {

    // 62. Unique paths
    // https://www.youtube.com/watch?v=4Zq2Fnd6tl0&list=PLtQWXpf5JNGKrsnLTGQ4C-EA9th4EJ7Th

    /*
     * |-------- // start->|1 1 1 1 1 // |1 2 3 4 5 // |1 3 6 10 15 <- end //
     * |_________ t-> O(mxn) s-> O(mxn)
     */
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0)
                    dp[i][j] = 1;
                else
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }
        return dp[m - 1][n - 1];
    }

    // 118. Pascal's Triangle
    // https://www.youtube.com/watch?v=VJBUH3chC64&list=PLtQWXpf5JNGKrsnLTGQ4C-EA9th4EJ7Th&index=2
    // t -> O(n^2) s -> O(n^2)

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> list = new ArrayList<>(Arrays.asList(1)); // jth start index
            for (int j = 1; j < i; j++) {
                List<Integer> prev = result.get(i - 1);
                list.add(prev.get(j - 1) + prev.get(j));
            }
            if (i > 0)
                list.add(1); // jth end index
            result.add(list);
        }
        return result;
    }

    // 64. Minimum Path Sum
    // https://www.youtube.com/watch?v=8-6MOl7p7Qs&list=PLtQWXpf5JNGKrsnLTGQ4C-EA9th4EJ7Th&index=3
    // t-> O(mxn) s-> O(1)

    public int minPathSum(int[][] grid) {
        int n = grid.length;
        int m = grid[0].length;

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int top = i - 1 < 0 ? Integer.MAX_VALUE : grid[i - 1][j];
                int bottom = j - 1 < 0 ? Integer.MAX_VALUE : grid[i][j - 1];
                int min = top == Integer.MAX_VALUE && bottom == Integer.MAX_VALUE ? 0 : Math.min(top, bottom);
                grid[i][j] += min;
            }
        }
        return grid[n - 1][m - 1];
    }

    // 329. Longest Increasing Path in a Matrix

    // Given an m x n integers matrix, return the length of the longest increasing
    // path in matrix.

    // From each cell, you can either move in four directions: left, right, up, or
    // down. You may not move diagonally or move outside the boundary (i.e.,
    // wrap-around is not allowed).

    // https://www.youtube.com/watch?v=uLjO2LUlLN4&list=PLtQWXpf5JNGKrsnLTGQ4C-EA9th4EJ7Th&index=4

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

    // 53. Maximum Subarray
    // Given an integer array nums, find the contiguous subarray (containing at
    // least one number) which has the largest sum and return its sum.

    // Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    // Output: 6
    // Explanation: [4,-1,2,1] has the largest sum = 6.

    // Kadane's Algorithm
    // t -> O(n) s -> O(1)

    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] = Math.max(nums[i], nums[i] + nums[i - 1]);
            max = Math.max(max, nums[i]);
        }
        return max;
    }

    // OR

    public static long maximumSubarrayValue(int[] ar) {

        if (ar == null || ar.length == 0)
            return 0L;
        int n = ar.length, maxValue, sum;

        maxValue = sum = ar[0];

        for (int i = 1; i < n; i++) {

            // At each step consider continuing the current subarray
            // or starting a new one because adding the next element
            // doesn't acutally make the subarray sum any better.
            if (ar[i] > sum + ar[i])
                sum = ar[i];
            else
                sum = sum + ar[i];

            if (sum > maxValue)
                maxValue = sum;
        }

        return maxValue;
    }

    // a tile can either hold 1 or 2 length bricks.
    // Form an input length by adding the tiles no of ways.
    public int possibleTileCombo(int input) {
        int[] dp = new int[input + 1];
        return dfspossibleTileCombo(input, dp);
    }

    // O(n), O(n)
    private int dfspossibleTileCombo(int n, int[] dp) {
        if (n < 0)
            return 0;
        if (n == 0)
            return 1;
        if (dp[n] > 0)
            return dp[n];
        dp[n] = dfspossibleTileCombo(n - 1, dp) + dfspossibleTileCombo(n - 2, dp);
        return dp[n];
    }

    // 0/1 Knapsack.
    // Find max value items that fits given capacity

    // https://www.youtube.com/watch?v=cJ21moQpofY

    public static int knapsack(int capacity, int[] W, int[] V) throws IllegalAccessException {
        if (W == null || V == null || W.length != V.length || capacity < 0)
            throw new IllegalAccessException("Invalid Input");

        final int N = W.length;

        // Initialize a table where individual rows represent items
        // and columns represent the weight of the knapsack
        int[][] dp = new int[N + 1][capacity + 1];

        for (int i = 1; i <= N; i++) {
            // Get the value and weigth of current item
            int w = W[i - 1];
            int v = V[i - 1];
            for (int j = 1; j <= capacity; j++) {
                // Consider not picking this element (initial best case)
                dp[i][j] = dp[i - 1][j];

                // consider including current element and see if this would be profitable.
                if (j >= w && dp[i - 1][j - w] + v > dp[i][j])
                    dp[i][j] = dp[i - 1][j - w] + v;
            }
        }

        int size = capacity;
        List<Integer> itemsSelected = new ArrayList<>();
        // Using the information inside the table we can backtrack and determine
        // which items were selected during the dynamic programming phase. The idea
        // is that if DP[i][sz] != DP[i-1][sz] then the item was selected
        for (int i = N; i > 0; i--) {
            if (dp[i][size] != dp[i - 1][size]) {
                int itemIndex = i - 1;
                itemsSelected.add(itemIndex);
                size -= W[itemIndex];
            }
        }

        // Return the items that were selected
        // java.util.Collections.reverse(itemsSelected);
        // return itemsSelected;

        // Return the maximum profit
        return dp[N][capacity];

    }

    public static int INF = Integer.MAX_VALUE;

    public static int coinChange(int[] coins, int amount) {

        if (coins == null)
            throw new IllegalArgumentException("Coins array is null");
        if (coins.length == 0)
            throw new IllegalArgumentException("No coin values :/");

        final int N = coins.length;
        // Initialize table and set first row to be infinity
        int[][] dp = new int[N + 1][amount + 1];
        java.util.Arrays.fill(dp[0], INF);
        dp[1][0] = 0;

        // Iterate through all the coins
        for (int i = 1; i <= N; i++) {
            int coinValue = coins[i - 1];
            for (int j = 1; j <= amount; j++) {

                // Consider not selecting this coin
                dp[i][j] = dp[i - 1][j];

                // Try selecting this coin if it's better
                if (j - coinValue >= 0 && dp[i][j - coinValue] + 1 < dp[i][j])
                    dp[i][j] = dp[i][j - coinValue] + 1;
            }
        }

        // The amount we wanted to make cannot be made :/
        if (dp[N][amount] == INF)
            return -1;

        // Return the minimum number of coins needed
        return dp[N][amount];
    }

    // 1143. Longest Common Subsequence

    // Given two strings text1 and text2, return the length of their longest common
    // subsequence. If there is no common subsequence, return 0.

    // Example 1:
    // Input: text1 = "abcde", text2 = "ace"
    // Output: 3
    // Explanation: The longest common subsequence is "ace" and its length is 3.

    // Example 2:
    // Input: text1 = "abc", text2 = "abc"
    // Output: 3
    // Explanation: The longest common subsequence is "abc" and its length is 3.

    // Example 3:
    // Input: text1 = "abc", text2 = "def"
    // Output: 0
    // Explanation: There is no such common subsequence, so the result is 0.

    public int longestCommonSubsequence(String text1, String text2) {
        if (text1 == null || text1.length() == 0 || text2 == null || text2.length() == 0)
            return 0;

        final int n = text1.length();
        final int m = text2.length();
        int[][] dp = new int[n + 1][m + 1];

        // Suppose A = a1a2..an-1an and B = b1b2..bn-1bn
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {

                // If ends match the LCS(a1a2..an-1an, b1b2..bn-1bn) = LCS(a1a2..an-1,
                // b1b2..bn-1) + 1
                if (text1.charAt(i - 1) == text2.charAt(j - 1))
                    dp[i][j] = dp[i - 1][j - 1] + 1;

                // If the ends do not match the LCS of a1a2..an-1an and b1b2..bn-1bn is
                // max( LCS(a1a2..an-1, b1b2..bn-1bn), LCS(a1a2..an-1an, b1b2..bn-1) )
                else
                    dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
            }
        }

        int lcsLen = dp[n][m];
        char[] lcs = new char[lcsLen];
        int index = 0;

        // Backtrack to find a LCS. We search for the cells
        // where we included an element which are those with
        // dp[i][j] != dp[i-1][j] and dp[i][j] != dp[i][j-1])
        int i = n, j = m;
        while (i >= 1 && j >= 1) {

            int v = dp[i][j];

            // The order of these may output different LCSs
            while (i > 1 && dp[i - 1][j] == v)
                i--;
            while (j > 1 && dp[i][j - 1] == v)
                j--;

            // Make sure there is a match before adding
            if (v > 0)
                lcs[lcsLen - index++ - 1] = text1.charAt(i - 1); // or B[j-1];

            i--;
            j--;
        }

        return new String(lcs, 0, lcsLen).length();

    }

    // 790. Domino and Tromino Tiling

    public static int num;
    static Long[][] dp;
    static long MOD = 1_000_000_007;

    public int numTilings(int N) {
        n = N;
        dp = new Long[n + 1][1 << 2];
        long ans = f(0, true, true);
        return (int) ans;
    }

    // t1 - whether tile 1 is available
    // t2 - whether tile 2 is available
    static long f(int i, boolean t1, boolean t2) {
        // Finished fully tiling the board.
        if (i == num) {
            return 1;
        }
        int state = makeState(t1, t2);
        if (dp[i][state] != null) {
            return dp[i][state];
        }

        // Zones that define which regions are free. For the surrounding 4 tiles:
        // t1 t3
        // t2 t4
        boolean t3 = i + 1 < num;
        boolean t4 = i + 1 < num;

        long count = 0;

        // Placing:
        // XX
        // X
        if (t1 && t2 && t3)
            count += f(i + 1, false, true);

        // Placing:
        // X
        // XX
        if (t1 && t2 && t4)
            count += f(i + 1, true, false);

        // Placing:
        // XX
        // #X
        if (t1 && !t2 && t3 && t4)
            count += f(i + 1, false, false);

        // Placing:
        // #X
        // XX
        if (!t1 && t2 && t3 && t4)
            count += f(i + 1, false, false);

        // Placing
        // X
        // X
        if (t1 && t2)
            count += f(i + 1, true, true);

        // Placing two horizontals. We don't place 2 verticals because
        // that's accounted for with the single vertical tile:
        // XX
        // XX
        if (t1 && t2 && t3 && t4)
            count += f(i + 1, false, false);

        // Placing:
        // XX
        // #
        if (t1 && !t2 && t3)
            count += f(i + 1, false, true);

        // Placing:
        // #
        // XX
        if (!t1 && t2 && t4)
            count += f(i + 1, true, false);

        // Current column is already fully tiled, so move to next column
        // #
        // #
        if (!t1 && !t2)
            count += f(i + 1, true, true);

        return dp[i][state] = count % MOD;
    }

    static int makeState(boolean row1, boolean row2) {
        int state = 0;
        if (row1)
            state |= 0b01;
        if (row2)
            state |= 0b10;
        return state;
    }

}
