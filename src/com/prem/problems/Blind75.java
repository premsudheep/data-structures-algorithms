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

// https://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU
// https://leetcode.com/list/xi4ci4ig/

public class Blind75 {

    // 1. Array:

    // 1. Two Sum
    // Given an array of integers nums and an integer target, return indices of the
    // two numbers such that they add up to target.

    // You may assume that each input would have exactly one solution, and you may
    // not use the same element twice.
    // You can return the answer in any order.

    // Input: nums = [2,7,11,15], target = 9
    // Output: [0,1]
    // Output: Because nums[0] + nums[1] == 9, we return [0, 1].

    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int difference = target - nums[i];
            if (map.containsKey(difference)) {
                result[0] = i;
                result[1] = map.get(difference);
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
    }

    // 121. Best Time to Buy and Sell Stock
    // You are given an array prices where prices[i] is the price of a given stock
    // on the ith day.
    // You want to maximize your profit by choosing a single day to buy one stock
    // and choosing a different day in the future to sell that stock.
    // Return the maximum profit you can achieve from this transaction. If you
    // cannot achieve any profit, return 0.

    // Input: prices = [7,1,5,3,6,4]
    // Output: 5
    // Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit =
    // 6-1 = 5.
    // Note that buying on day 2 and selling on day 1 is not allowed because you
    // must buy before you sell.

    public int maxProfit(int[] prices) {
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i] < min)
                min = prices[i];
            else
                max = Math.max(max, prices[i] - min);
        }
        return max;
    }

    // 217. Contains Duplicate
    // Given an integer array nums, return true if any value appears at least twice
    // in the array, and return false if every element is distinct.

    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num))
                return true;
            else
                set.add(num);
        }
        return false;
    }

    // 238. Product of Array Except Self

    // Given an integer array nums, return an array answer such that answer[i] is
    // equal to the product of all the elements of nums except nums[i]. Should run
    // in O(n)

    public int[] productExceptSelf1(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];
        int[] left = new int[length];
        int[] right = new int[length];

        left[0] = 1;
        for (int i = 1; i < length; i++) {
            left[i] = left[i - 1] * nums[i - 1];
        }

        right[length - 1] = 1;
        for (int i = length - 2; i >= 0; i--) {
            right[i] = right[i + 1] * nums[i + 1];
        }

        for (int i = 0; i < length; i++) {
            answer[i] = left[i] * right[i];
        }

        return answer;
    }

    // Solving above in O(1) space
    public int[] productExceptSelf2(int[] nums) {
        int length = nums.length;
        int[] answer = new int[length];

        answer[0] = 1;
        for (int i = 1; i < length; i++) {
            answer[i] = answer[i - 1] * nums[i - 1];
        }

        int right = 1;
        for (int i = length - 1; i >= 0; i--) {
            answer[i] = answer[i] * right;
            right *= nums[i];
        }

        return answer;
    }

    // 53. Maximum Subarray
    // Given an integer array nums, find the contiguous subarray (containing at
    // least one number) which has the largest sum and return its sum.

    // Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
    // Output: 6
    // Explanation: [4,-1,2,1] has the largest sum = 6.

    // Input: nums = [5,4,-1,7,8]
    // Output: 23

    public int maxSubArray(int[] nums) {
        int max = nums[0];
        for (int i = 1; i < nums.length; i++) {
            nums[i] = Math.max(nums[i], nums[i] + nums[i - 1]);
            max = Math.max(max, nums[i]);
        }
        return max;
    }

    // 152. Maximum Product Subarray

    // Given an integer array nums, find a contiguous non-empty subarray within the
    // array that has the largest product, and return the product.

    public int maxProduct(int[] nums) {
        if (nums.length == 0)
            return 0;

        int maxSoFar = nums[0];
        int minSoFar = nums[0];
        int result = maxSoFar;

        for (int i = 1; i < nums.length; i++) {
            int curr = nums[i];
            int tempMax = Math.max(curr, Math.max(maxSoFar * curr, minSoFar * curr));
            minSoFar = Math.min(curr, Math.min(maxSoFar * curr, minSoFar * curr));
            maxSoFar = tempMax;
            result = Math.max(maxSoFar, result);
        }

        return result;
    }

    // 153. Find Minimum in Rotated Sorted Array

    // Suppose an array of length n sorted in ascending order is rotated between 1
    // and n times. For example, the array nums = [0,1,2,4,5,6,7] might become:

    // [4,5,6,7,0,1,2] if it was rotated 4 times.
    // [0,1,2,4,5,6,7] if it was rotated 7 times.
    // Notice that rotating an array [a[0], a[1], a[2], ..., a[n-1]] 1 time results
    // in the array [a[n-1], a[0], a[1], a[2], ..., a[n-2]].

    // Given the sorted rotated array nums of unique elements, return the minimum
    // element of this array.

    // You must write an algorithm that runs in O(log n) time.

    // Example 1:
    // Input: nums = [3,4,5,1,2]
    // Output: 1
    // Explanation: The original array was [1,2,3,4,5] rotated 3 times.

    // Example 2:
    // Input: nums = [4,5,6,7,0,1,2]
    // Output: 0
    // Explanation: The original array was [0,1,2,4,5,6,7] and it was rotated 4
    // times.

    // Example 3:
    // Input: nums = [11,13,15,17]
    // Output: 11
    // Explanation: The original array was [11,13,15,17] and it was rotated 4 times.

    public int findMin(int[] nums) {

        // If the list has just one element then return that element.
        if (nums.length == 1)
            return nums[0];

        // initialize left and right pointers.
        int left = 0, right = nums.length - 1;

        // If the last element is greater than the first element then there is no
        // rotation.
        // e.g. 1 < 2 < 3 < 4 < 5 < 7. Already sorted array.
        if (nums[right] > nums[0])
            return nums[0];

        // Binary search way until
        // nums[mid] > nums[mid + 1] Hence, mid+1 is the smallest.
        // nums[mid - 1] > nums[mid] Hence, mid is the smallest.

        while (right >= left) {
            // Find the mid element
            int mid = left + (right - left) / 2;

            // if mid element is greater than its next element then mid + 1 element is the
            // smallest
            // This point would be the point of change. From higher to lower value.
            if (nums[mid] > nums[mid + 1])
                return nums[mid + 1];

            // if mid element is less than its previous element then mid element is the
            // smallest
            if (nums[mid - 1] > nums[mid])
                return nums[mid];

            // if mid elements value is greater then th 0th element this means
            // the least value is somewhere to the right as we are still dealing with
            // elements.
            if (nums[mid] > nums[0])
                left = mid + 1;
            else
                // if nums[0] is greater than the mid value then this means the smallest value
                // is somewhere to the left
                right = mid - 1;
        }
        return -1;
    }

    // 33. Search in Rotated Sorted Array

    // There is an integer array nums sorted in ascending order (with distinct
    // values).

    // Prior to being passed to your function, nums is possibly rotated at an
    // unknown pivot index k (1 <= k < nums.length) such that the resulting array is
    // [nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]
    // (0-indexed). For example, [0,1,2,4,5,6,7] might be rotated at pivot index 3
    // and become [4,5,6,7,0,1,2].

    // Given the array nums after the possible rotation and an integer target,
    // return the index of target if it is in nums, or -1 if it is not in nums.

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

    // 15. 3Sum

    // Given an integer array nums, return all the triplets [nums[i], nums[j],
    // nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] +
    // nums[k] == 0.
    // Input: nums = [-1,0,1,2,-1,-4]
    // Output: [[-1,-1,2],[-1,0,1]]

    // two pointer solution

    public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        List<List<Integer>> result = new ArrayList<>();
        for (int i = 0; i < nums.length && nums[i] <= 0; i++) {
            if (i == 0 || nums[i - 1] != nums[i]) {
                twoSum(nums, i, result);
            }
        }
        return result;
    }

    private void twoSum(int[] nums, int i, List<List<Integer>> result) {
        int lo = i + 1, hi = nums.length - 1;
        while (lo < hi) {
            int sum = nums[i] + nums[lo] + nums[hi];
            if (sum < 0)
                lo++;
            else if (sum > 0)
                hi--;
            else {
                result.add(Arrays.asList(nums[i], nums[lo++], nums[hi--]));
                while (lo < hi && nums[lo] == nums[lo - 1])
                    lo++;
            }
        }
    }

    // 11. Container With Most Water

    public int maxArea(int[] height) {
        int max = Integer.MIN_VALUE;
        int i = 0;
        int j = height.length - 1;
        while (i < j) {
            int min = Math.min(height[i], height[j]);
            max = Math.max(max, min * (j - i));
            if (height[i] < height[j])
                i++;
            else
                j--;
        }
        return max;
    }

    // 2. Binary

    // 371. Sum of Two Integers
    // Given two integers a and b, return the sum of the two integers without using
    // the operators + and -.

    public int getSum(int a, int b) {
        while (b != 0) {
            int answer = a ^ b;
            int carry = (a & b) << 1;
            a = answer;
            b = carry;
        }
        return a;
    }

    // 191. Number of 1 Bits
    // Write a function that takes an unsigned integer and returns the number of '1'
    // bits it has (also known as the Hamming weight).

    public int hammingWeight1(int n) {
        return Integer.bitCount(n);
    }

    public int hammingWeight2(int n) {
        int bits = 0;
        int mask = 1; // -> 1 followed by 31 0's
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

    // 338. Counting Bits

    // Given an integer n, return an array ans of length n + 1 such that for each i
    // (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.

    // Example 1:

    // Input: n = 2
    // Output: [0,1,1]
    // Explanation:
    // 0 --> 0
    // 1 --> 1
    // 2 --> 10

    // Example 2:

    // Input: n = 5
    // Output: [0,1,1,2,1,2]
    // Explanation:
    // 0 --> 0
    // 1 --> 1
    // 2 --> 10
    // 3 --> 11
    // 4 --> 100
    // 5 --> 101
    public int[] countBits(int n) {
        int[] ans = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            ans[i] = hammingWeight3(i);
        }
        return ans;
    }

    private int hammingWeight3(int n) {
        int bits = 0;
        int mask = 1;
        for (int i = 0; i < 32; i++) {
            if ((n & mask) != 0) {
                bits++;
            }
            mask <<= 1;
        }
        return bits;
    }

    // 268. Missing Number

    // Input: nums = [3,0,1]
    // Output: 2

    // Input: nums = [9,6,4,2,3,5,7,0,1]
    // Output: 8

    public int missingNumber1(int[] nums) {
        int sum = 0;
        for (int i : nums) {
            sum += i;
        }

        int n = nums.length + 1;
        return (n * (n - 1)) / 2 - sum;
    }

    // OR

    public int missingNumber2(int[] nums) {
        int missing = nums.length;
        for (int i = 0; i < nums.length; i++) {
            missing ^= i ^ nums[i];
        }
        return missing;
    }

    // 190. Reverse Bits

    // Reverse bits of a given 32 bits unsigned integer.

    public int reverseBits(int n) {
        int ans = 0;
        for (int i = 0; i < 32; i++) {
            ans <<= 1;
            ans = ans | (n & 1);
            n >>= 1;
        }
        return ans;
    }

    // 3. Dynamic Programming

    // 70. Climbing Stairs

    // You are climbing a staircase. It takes n steps to reach the top.
    // Each time you can either climb 1 or 2 steps. In how many distinct ways can
    // you climb to the top?

    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    // 322. Coin Change

    // You are given an integer array coins representing coins of different
    // denominations and an integer amount representing a total amount of money.
    // Return the fewest number of coins that you need to make up that amount. If
    // that amount of money cannot be made up by any combination of the coins,
    // return -1.
    // You may assume that you have an infinite number of each kind of coin.

    // Input: coins = [1,2,5], amount = 11
    // Output: 3
    // Explanation: 11 = 5 + 5 + 1

    private static final int INF = Integer.MAX_VALUE;

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

    // 300. Longest Increasing Subsequence

    // Given an integer array nums, return the length of the longest strictly
    // increasing subsequence.

    // A subsequence is a sequence that can be derived from an array by deleting
    // some or no elements without changing the order of the remaining elements. For
    // example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].

    // Example 1:
    // Input: nums = [10,9,2,5,3,7,101,18]
    // Output: 4
    // Explanation: The longest increasing subsequence is [2,3,7,101], therefore the
    // length is 4.

    // Example 2:
    // Input: nums = [0,1,0,3,2,3]
    // Output: 4

    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
        }

        int longest = 0;
        for (int c : dp) {
            longest = Math.max(longest, c);
        }

        return longest;
    }

    // 1143. Longest Common Subsequence

    // https://www.youtube.com/watch?v=NnD96abizww

    // Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.

    // A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

    // For example, "ace" is a subsequence of "abcde".
    // A common subsequence of two strings is a subsequence that is common to both strings.

    

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

    // 4. Graph

    // 133. Clone Graph

    // Given a reference of a node in a connected undirected graph. Return a deep
    // copy (clone) of the graph.

    // Each node in the graph contains a value (int) and a list (List[Node]) of its
    // neighbors.

    // class Node {
    // public int val;
    // public List<Node> neighbors;
    // }

    // Test case format:

    // For simplicity, each node's value is the same as the node's index
    // (1-indexed). For example, the first node with val == 1, the second node with
    // val == 2, and so on. The graph is represented in the test case using an
    // adjacency list.

    // An adjacency list is a collection of unordered lists used to represent a
    // finite graph. Each list describes the set of neighbors of a node in the
    // graph.

    // The given node will always be the first node with val = 1. You must return
    // the copy of the given node as a reference to the cloned graph.

    // Input: adjList = [[2,4],[1,3],[2,4],[1,3]]
    // Output: [[2,4],[1,3],[2,4],[1,3]]
    // Explanation: There are 4 nodes in the graph.
    // 1st node (val = 1)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
    // 2nd node (val = 2)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).
    // 3rd node (val = 3)'s neighbors are 2nd node (val = 2) and 4th node (val = 4).
    // 4th node (val = 4)'s neighbors are 1st node (val = 1) and 3rd node (val = 3).

    // DFS
    private Map<Node, Node> visited = new HashMap<>();

    public Node cloneGraph1(Node node) {
        if (node == null)
            return node;

        // already visited before, then return clone from the visited
        if (visited.containsKey(node))
            return visited.get(node);

        // create a clone for given node and iterate to clone its neighbors.
        Node cloneNode = new Node(node.val, new ArrayList<>());
        visited.put(node, cloneNode);
        for (Node neigh : node.neighbors) {
            cloneNode.neighbors.add(cloneGraph1(neigh));
        }
        return cloneNode;
    }

    // BFS
    public Node cloneGraph2(Node node) {
        if (node == null) {
            return node;
        }

        // Hash map to save the visited node and it's respective clone
        // as key and value respectively. This helps to avoid cycles.
        HashMap<Node, Node> visited = new HashMap<>();

        // Put the first node in the queue
        LinkedList<Node> queue = new LinkedList<Node>();
        queue.add(node);
        // Clone the node and put it in the visited dictionary.
        visited.put(node, new Node(node.val, new ArrayList<>()));

        // Start BFS traversal
        while (!queue.isEmpty()) {
            // Pop a node say "n" from the from the front of the queue.
            Node n = queue.remove();
            // Iterate through all the neighbors of the node "n"
            for (Node neighbor : n.neighbors) {
                if (!visited.containsKey(neighbor)) {
                    // Clone the neighbor and put in the visited, if not present already
                    visited.put(neighbor, new Node(neighbor.val, new ArrayList<>()));
                    // Add the newly encountered node to the queue.
                    queue.add(neighbor);
                }
                // Add the clone of the neighbor to the neighbors of the clone node "n".
                visited.get(n).neighbors.add(visited.get(neighbor));
            }
        }

        // Return the clone of the node from visited.
        return visited.get(node);
    }

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

    // 207. Course Schedule

    // There are a total of numCourses courses you have to take, labeled from 0 to
    // numCourses - 1. You are given an array prerequisites where prerequisites[i] =
    // [ai, bi] indicates that you must take course bi first if you want to take
    // course ai.
    // For example, the pair [0, 1], indicates that to take course 0 you have to
    // first take course 1.
    // Return true if you can finish all courses. Otherwise, return false.

    // Example 1:
    // Input: numCourses = 2, prerequisites = [[1,0]]
    // Output: true
    // Explanation: There are a total of 2 courses to take.
    // To take course 1 you should have finished course 0. So it is possible.

    // Example 2:
    // Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
    // Output: false
    // Explanation: There are a total of 2 courses to take.
    // To take course 1 you should have finished course 0, and to take course 0 you
    // should also have finished course 1. So it is impossible.

    // Kahn's Algorithm

    public boolean canFinish(int numCourses, int[][] prerequisites) {
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

            // record in-degree of each vertex
            inDegree[dest]++;
        }

        Queue<Integer> q = new LinkedList<>();
        for (int i = 0; i < inDegree.length; i++) {
            if (inDegree[i] == 0)
                q.add(i);
        }

        int index = 0;
        // process until the q becomes empty
        // Breadth-First Search
        while (!q.isEmpty()) {
            int current = q.remove();
            topologicalOrder[index++] = current;

            // Reduce the in-degree of each neighbor by 1
            if (adjList.containsKey(current)) {
                for (int neigh : adjList.get(current)) {
                    inDegree[neigh]--;

                    // if in-degree of a neighbor is 0, then add it to the q.
                    if (inDegree[neigh] == 0)
                        q.add(neigh);
                }
            }
        }

        if (index == numCourses)
            return true;

        return false;

    }

    // 417. Pacific Atlantic Water Flow

    // https://www.youtube.com/watch?v=krL3r7MY7Dc

    // There is an m x n rectangular island that borders both the Pacific Ocean and
    // Atlantic Ocean. The Pacific Ocean touches the island's left and top edges,
    // and the Atlantic Ocean touches the island's right and bottom edges.

    // The island is partitioned into a grid of square cells. You are given an m x n
    // integer matrix heights where heights[r][c] represents the height above sea
    // level of the cell at coordinate (r, c).

    // The island receives a lot of rain, and the rain water can flow to neighboring
    // cells directly north, south, east, and west if the neighboring cell's height
    // is less than or equal to the current cell's height. Water can flow from any
    // cell adjacent to an ocean into the ocean.

    // Return a 2D list of grid coordinates result where result[i] = [ri, ci]
    // denotes that rain water can flow from cell (ri, ci) to both the Pacific and
    // Atlantic oceans.

    // Example 1:
    // Input: heights =
    // [[1,2,2,3,5],[3,2,3,4,4],[2,4,5,3,1],[6,7,1,4,5],[5,1,1,2,4]]
    // Output: [[0,4],[1,3],[1,4],[2,2],[3,0],[3,1],[4,0]]

    // Example 2:
    // Input: heights = [[2,1],[1,2]]
    // Output: [[0,0],[0,1],[1,0],[1,1]]

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

    // 200. Number of Islands

    // Given an m x n 2D binary grid grid which represents a map of '1's (land) and
    // '0's (water), return the number of islands.

    // An island is surrounded by water and is formed by connecting adjacent lands
    // horizontally or vertically. You may assume all four edges of the grid are all
    // surrounded by water.

    // Example 1:

    // Input: grid = [
    // ["1","1","1","1","0"],
    // ["1","1","0","1","0"],
    // ["1","1","0","0","0"],
    // ["0","0","0","0","0"]
    // ]
    // Output: 1

    // Example 2:

    // Input: grid = [
    // ["1","1","0","0","0"],
    // ["1","1","0","0","0"],
    // ["0","0","1","0","0"],
    // ["0","0","0","1","1"]
    // ]
    // Output: 3

    public int numIslands(char[][] grid) {
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

    // 128. Longest Consecutive Sequence

    // Given an unsorted array of integers nums, return the length of the longest
    // consecutive elements sequence.

    // You must write an algorithm that runs in O(n) time.

    // Example 1:
    // Input: nums = [100,4,200,1,3,2]
    // Output: 4
    // Explanation: The longest consecutive elements sequence is [1, 2, 3, 4].
    // Therefore its length is 4.

    // Example 2:
    // Input: nums = [0,3,7,2,5,8,4,6,0,1]
    // Output: 9

    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            set.add(num);
        }

        int longestStreak = 0;

        for (int num : set) {
            // Since setstore values in ascending order.
            if (!set.contains(num - 1)) {
                int currentNum = num;
                int currentStreak = 1;

                while (set.contains(currentNum + 1)) {
                    currentNum += 1;
                    currentStreak += 1;
                }

                longestStreak = Math.max(longestStreak, currentStreak);
            }
        }

        return longestStreak;
    }

    // 269. Alien Dictionary

    // here is a new alien language that uses the English alphabet. However, the
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

    // Example 1:
    // Input: words = ["wrt","wrf","er","ett","rftt"]
    // Output: "wertf"

    // Example 2:
    // Input: words = ["z","x"]
    // Output: "zx"

    // Example 3:
    // Input: words = ["z","x","z"]
    // Output: ""
    // Explanation: The order is invalid, so return "".

    public String alienOrder(String[] words) {
        // Step 0: Create data structures and find all unique letters.
        Map<Character, List<Character>> adjList = new HashMap<>();
        Map<Character, Integer> inDegree = new HashMap<>();

        for (String s : words) {
            for (char c : s.toCharArray()) {
                inDegree.put(c, 0);
                adjList.put(c, new ArrayList<>());
            }
        }

        // Step 1: Find the edges
        for (int i = 0; i < words.length - 1; i++) {
            String word1 = words[i];
            String word2 = words[i + 1];
            // Check word2 is not a prefix of word1.
            if (word1.length() > word2.length() && word1.startsWith(word2)) {
                return "";
            }

            // Find the first non match and insert the corresponding relation.
            for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) {
                char ch1 = word1.charAt(j);
                char ch2 = word2.charAt(j);
                if (ch1 != ch2) {
                    adjList.get(ch1).add(ch2);
                    inDegree.put(ch2, inDegree.get(ch2) + 1);
                    break;
                }
            }
        }

        // Step 2: Breadth-First Search
        StringBuilder sb = new StringBuilder();
        Queue<Character> q = new LinkedList<>();
        for (char ch : inDegree.keySet()) {
            if (inDegree.get(ch).equals(0)) {
                q.add(ch);
            }
        }

        while (!q.isEmpty()) {
            char current = q.remove();
            sb.append(current);
            for (char next : adjList.get(current)) {
                inDegree.put(next, inDegree.get(next) - 1);
                if (inDegree.get(next).equals(0)) {
                    q.add(next);
                }
            }
        }

        if (sb.length() < inDegree.size()) {
            return "";
        }

        return sb.toString();
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

    // 323. Number of Connected Components in an Undirected Graph

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

    // 5. Interval

    // 57. Insert Interval

    // You are given an array of non-overlapping intervals intervals where
    // intervals[i] = [starti, endi] represent the start and the end of the ith
    // interval and intervals is sorted in ascending order by starti. You are also
    // given an interval newInterval = [start, end] that represents the start and
    // end of another interval.

    // Insert newInterval into intervals such that intervals is still sorted in
    // ascending order by starti and intervals still does not have any overlapping
    // intervals (merge overlapping intervals if necessary).

    // Return intervals after the insertion.

    // Example 1:
    // Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
    // Output: [[1,5],[6,9]]

    // Example 2:
    // Input: intervals = [[1,2],[3,5],[6,7],[8,10],[12,16]], newInterval = [4,8]
    // Output: [[1,2],[3,10],[12,16]]
    // Explanation: Because the new interval [4,8] overlaps with [3,5],[6,7],[8,10].

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

    // 56. Merge Intervals

    // Given an array of intervals where intervals[i] = [starti, endi], merge all
    // overlapping intervals, and return an array of the non-overlapping intervals
    // that cover all the intervals in the input.

    // Example 1:
    // Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    // Output: [[1,6],[8,10],[15,18]]
    // Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].

    // Example 2:
    // Input: intervals = [[1,4],[4,5]]
    // Output: [[1,5]]
    // Explanation: Intervals [1,4] and [4,5] are considered overlapping.

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

    // 435. Non-overlapping Intervals

    // Given an array of intervals intervals where intervals[i] = [starti, endi],
    // return the minimum number of intervals you need to remove to make the rest of
    // the intervals non-overlapping.

    // Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
    // Output: 1
    // Explanation: [1,3] can be removed and the rest of the intervals are
    // non-overlapping.

    public int eraseOverlapIntervals(int[][] intervals) {
        if (intervals.length == 0)
            return 0;

        Arrays.sort(intervals, (a, b) -> a[1] - b[1]);

        int prev = 0, count = 0;
        for (int i = 1; i < intervals.length; i++) {
            if (intervals[prev][1] > intervals[i][0]) {
                if (intervals[prev][1] > intervals[i][1]) {
                    prev = i;
                }
                count++;
            } else {
                prev = i;
            }
        }
        return count;
    }

    // 252. Meeting Rooms
    // Given an array of meeting time intervals where intervals[i] = [starti, endi],
    // determine if a person could attend all meetings.

    // Example 1:
    // Input: intervals = [[0,30],[5,10],[15,20]]
    // Output: false

    // Example 2:
    // Input: intervals = [[7,10],[2,4]]
    // Output: true

    public boolean canAttendMeetings(int[][] intervals) {
        int[] start = new int[intervals.length];
        int[] end = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            start[i] = intervals[i][0];
            end[i] = intervals[i][1];
        }
        Arrays.sort(start);
        Arrays.sort(end);
        for (int i = 0; i < start.length - 1; i++) {
            if (start[i + 1] < end[i])
                return false;
        }
        return true;
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

    // 6. Linked List

    public class ListNode {
        int val;
        ListNode next;

        ListNode() {
        }

        ListNode(int val) {
            this.val = val;
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }
    }

    // 206. Reverse Linked List

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        ListNode next = null;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }

    // 141. Linked List Cycle

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

    // 21. Merge Two Sorted Lists

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode result = new ListNode(-1);
        ListNode prev = result;

        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                prev.next = l1;
                l1 = l1.next;
            } else {
                prev.next = l2;
                l2 = l2.next;
            }
            prev = prev.next;
        }

        prev.next = l1 != null ? l1 : l2;
        return result.next;
    }

    // 23. Merge k Sorted Lists

    // You are given an array of k linked-lists lists, each linked-list is sorted in
    // ascending order.

    // Merge all the linked-lists into one sorted linked-list and return it.

    // Example 1:

    // Input: lists = [[1,4,5],[1,3,4],[2,6]]
    // Output: [1,1,2,3,4,4,5,6]
    // Explanation: The linked-lists are:
    // [
    // 1->4->5,
    // 1->3->4,
    // 2->6
    // ]
    // merging them into one sorted list:
    // 1->1->2->3->4->4->5->6

    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (ListNode node : lists) {
            while (node != null) {
                minHeap.add(node.val);
                node = node.next;
            }
        }

        ListNode last = new ListNode();
        ListNode first = last;
        while (!minHeap.isEmpty()) {
            last.next = new ListNode(minHeap.remove());
            last = last.next;
        }
        return first.next;
    }

    // 19. Remove Nth Node From End of List

    // Given the head of a linked list, remove the nth node from the end of the list
    // and return its head.

    // Input: head = [1,2,3,4,5], n = 2
    // Output: [1,2,3,5]

    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode first = dummy;
        ListNode second = dummy;
        for (int i = 1; i <= n + 1; i++) {
            first = first.next;
        }
        while (first != null) {
            first = first.next;
            second = second.next;
        }

        second.next = second.next.next;
        return dummy.next;
    }

    // 143. Reorder List

    // You are given the head of a singly linked-list. The list can be represented
    // as:

    // L0 → L1 → … → Ln - 1 → Ln
    // Reorder the list to be on the following form:

    // L0 → Ln → L1 → Ln - 1 → L2 → Ln - 2 → …
    // You may not modify the values in the list's nodes. Only nodes themselves may
    // be changed.

    public void reorderList(ListNode head) {
        // 1. Find the middle node.
        // 2. reverse the second part of the list.
        // 3. merge two sorted list.

        if (head == null)
            return;

        // Find the middle node of the linked list
        ListNode slow = head, fast = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
        }

        // reverse the second half in-place
        ListNode prev = null, next = null, curr = slow;
        while (curr != null) {
            next = curr.next;
            curr.next = prev;
            prev = curr;
            curr = next;
        }

        // merge 1->2->3->4 and 6->5->4 into 1->6->2->5->3->4
        ListNode first = head, second = prev;

        while (second.next != null) {
            next = first.next;
            first.next = second;
            first = next;

            next = second.next;
            second.next = first;
            second = next;
        }
    }

    // 7. Matrix

    // 73. Set Matrix Zeroes

    // Given an m x n integer matrix matrix, if an element is 0, set its entire row
    // and column to 0's, and return the matrix.

    // Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
    // Output: [[1,0,1],[0,0,0],[1,0,1]]

    // Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
    // Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

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

    // 54. Spiral Matrix

    // Given an m x n matrix, return all elements of the matrix in spiral order.

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

    // 48. Rotate Image

    // You are given an n x n 2D matrix representing an image, rotate the image by
    // 90 degrees (clockwise).

    // Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
    // Output: [[7,4,1],[8,5,2],[9,6,3]]

    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < (n + 1) / 2; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[n - 1 - j][i];
                matrix[n - 1 - j][i] = matrix[n - 1 - i][n - j - 1];
                matrix[n - 1 - i][n - j - 1] = matrix[j][n - 1 - i];
                matrix[j][n - 1 - i] = matrix[i][j];
                matrix[i][j] = temp;
            }
        }
    }

    // 79. Word Search

    // Given an m x n grid of characters board and a string word, return true if
    // word exists in the grid.

    // The word can be constructed from letters of sequentially adjacent cells,
    // where adjacent cells are horizontally or vertically neighboring. The same
    // letter cell may not be used more than once.

    // Example 1:

    // Input: board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]], word
    // = "ABCCED"
    // Output: true

    private char[][] board;
    private int ROWS;
    private int COLS;

    public boolean exist(char[][] board, String word) {
        this.board = board;
        this.ROWS = board.length;
        this.COLS = board[0].length;
        for (int row = 0; row < this.ROWS; row++)
            for (int col = 0; col < this.COLS; col++)
                if (this.backtrackWordExist(row, col, word, 0))
                    return true;
        return false;
    }

    private boolean backtrackWordExist(int row, int col, String word, int wordIndex) {
        // Step 1. check the bottom case.
        if (wordIndex >= word.length())
            return true;

        // Step 2. Check the boundaries.
        if (row < 0 || row == this.ROWS || col < 0 || col == this.COLS
                || this.board[row][col] != word.charAt(wordIndex))
            return false;

        // Step 3. explore the neighbors in DFS.
        boolean result = false;
        // mark the path before the next exploration
        this.board[row][col] = '#';
        int[] rowOffsets = { 0, 1, 0, -1 };
        int[] colOffsets = { 1, 0, -1, 0 };
        for (int d = 0; d < 4; d++) {
            result = this.backtrackWordExist(row + rowOffsets[d], col + colOffsets[d], word, wordIndex + 1);
            if (result)
                break;
        }

        // Step 4. clean up and return the result.
        this.board[row][col] = word.charAt(wordIndex);
        return result;
    }

    // 8. String

    // 3. Longest Substring Without Repeating Characters

    // Given a string s, find the length of the longest substring without repeating
    // characters.

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
    // Notice that the answer must be a substring, "pwke" is a subsequence and not a
    // substring.

    public int lengthOfLongestSubString(String s) {
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

    // 424. Longest Repeating Character Replacement

    // You are given a string s and an integer k. You can choose any character of
    // the string and change it to any other uppercase English character. You can
    // perform this operation at most k times.

    // Return the length of the longest substring containing the same letter you can
    // get after performing the above operations.

    // Example 1:

    // Input: s = "ABAB", k = 2
    // Output: 4
    // Explanation: Replace the two 'A's with two 'B's or vice versa.

    public int characterReplacement(String s, int k) {
        int[] cArr = new int[26];
        int maxCount = 0;
        int maxSize = 0;
        int start = 0;

        for (int end = 0; end < s.length(); end++) {
            cArr[s.charAt(end) - 'A']++;
            maxCount = Math.max(maxCount, cArr[s.charAt(end) - 'A']);

            // if max character frequency + distance between start and end is greater than k
            // this means we have considered changing more than k characters. so time to
            // shrink window.
            if (end - start + 1 - maxCount > k) {
                cArr[s.charAt(start) - 'A']--;
                start++;
            }
            maxSize = Math.max(maxSize, end - start + 1);
        }
        return maxSize;
    }

    // 76. Minimum Window Substring

    // Given two strings s and t of lengths m and n respectively, return the minimum
    // window substring of s such that every character in t (including duplicates)
    // is included in the window. If there is no such substring, return the empty
    // string "".

    // The testcases will be generated such that the answer is unique.

    // A substring is a contiguous sequence of characters within the string.

    // Example 1:

    // Input: s = "ADOBECODEBANC", t = "ABC"
    // Output: "BANC"
    // Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C'
    // from string t.

    public String minWindow(String s, String t) {
        if (s.length() == 0 || t.length() == 0)
            return "";

        // Dictionary which keeps a count of all the unique characters in t.
        Map<Character, Integer> dictT = new HashMap<>();
        for (int i = 0; i < t.length(); i++) {
            dictT.put(t.charAt(i), dictT.getOrDefault(t.charAt(i), 0) + 1);
        }

        // Number of unique charaters in t, which need to be present in the desired
        // window.
        int required = dictT.size();

        // left and right pointer
        int l = 0, r = 0;

        // formed is used to keep track of how many unique characters in t
        // are present in the current window in its desired frequency.
        // e.g. if t is "AABC" then the window must have two A's, one B and one C.

        // Thus formed would be = 3 when all these conditions are met.
        int formed = 0;

        Map<Character, Integer> windowCounts = new HashMap<Character, Integer>();

        // ans list of the form (window length, left, right)
        int[] ans = { -1, 0, 0 };

        while (r < s.length()) {
            // Add one character from the right to the window
            char c = s.charAt(r);
            int count = windowCounts.getOrDefault(c, 0);
            windowCounts.put(c, count + 1);

            // if the frequency of the current character added equals to the
            // desired count in t then increment the formed count by 1.
            if (dictT.containsKey(c) && windowCounts.get(c).intValue() == dictT.get(c).intValue())
                formed++;

            // Try and contract the window till the point where it ceases to be 'desirable'.
            while (l <= r && formed == required) {
                c = s.charAt(l);
                // Save the smallest window until now.
                if (ans[0] == -1 || r - l + 1 < ans[0]) {
                    ans[0] = r - l + 1;
                    ans[1] = l;
                    ans[2] = r;
                }

                // The character at the position pointed by the
                // `Left` pointer is no longer a part of the window.
                windowCounts.put(c, windowCounts.get(c) - 1);
                if (dictT.containsKey(c) && windowCounts.get(c).intValue() < dictT.get(c).intValue()) {
                    formed--;
                }
                // Move the left pointer ahead, this would help to look for a new window.
                l++;
            }
            // Keep expanding the window once we are done contracting.
            r++;
        }

        return ans[0] == -1 ? "" : s.substring(ans[1], ans[2] + 1);
    }

    // 242. Valid Anagram

    // Given two strings s and t, return true if t is an anagram of s, and false
    // otherwise.

    // Example 1:

    // Input: s = "anagram", t = "nagaram"
    // Output: true

    public boolean isAnagram(String s, String t) {
        if (s.length() != t.length())
            return false;

        int[] chars = new int[26];
        for (int i = 0; i < s.length(); i++) {
            chars[s.charAt(i) - 'a']++;
            chars[t.charAt(i) - 'a']--;
        }

        for (int i : chars) {
            if (i != 0)
                return false;
        }
        return true;
    }

    // 49. Group Anagrams
    // Given an array of strings strs, group the anagrams together. You can return
    // the answer in any order.

    // An Anagram is a word or phrase formed by rearranging the letters of a
    // different word or phrase, typically using all the original letters exactly
    // once.

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

    // 20. Valid Parentheses

    // Given a string s containing just the characters '(', ')', '{', '}', '[' and
    // ']', determine if the input string is valid.

    // An input string is valid if:

    // Open brackets must be closed by the same type of brackets.
    // Open brackets must be closed in the correct order.

    // Example 1:
    // Input: s = "()"
    // Output: true

    // Example 2:
    // Input: s = "()[]{}"
    // Output: true

    // Example 3:
    // Input: s = "(]"
    // Output: false

    public boolean isValid(String s) {
        Stack<Character> st = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch == '(' || ch == '{' || ch == '[')
                st.push(ch);
            else if (ch == ')' && !st.isEmpty() && st.peek() == '(')
                st.pop();
            else if (ch == '}' && !st.isEmpty() && st.peek() == '{')
                st.pop();
            else if (ch == ']' && !st.isEmpty() && st.peek() == '[')
                st.pop();
            else
                return false;
        }

        return st.isEmpty();
    }

    // 125. Valid Palindrome

    // Given a string s, determine if it is a palindrome, considering only
    // alphanumeric characters and ignoring cases.

    // Example 1:
    // Input: s = "A man, a plan, a canal: Panama"
    // Output: true
    // Explanation: "amanaplanacanalpanama" is a palindrome.

    public boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while (i < j) {
            while (i < j && !Character.isLetterOrDigit(s.charAt(i))) {
                i++;
            }
            while (i < j && !Character.isLetterOrDigit(s.charAt(j))) {
                j--;
            }
            if (i < j && Character.toLowerCase(s.charAt(i++)) != Character.toLowerCase(s.charAt(j--)))
                return false;
        }
        return true;
    }

    // 5. Longest Palindromic Substring
    // Given a string s, return the longest palindromic substring in s.

    // Example 1:

    // Input: s = "babad"
    // Output: "bab"
    // Note: "aba" is also a valid answer.

    public String longestPalindrome(String s) {
        if (s == null || s.length() == 0)
            return s;

        int len = s.length();
        String ans = "";
        int max = 0;
        boolean[][] dp = new boolean[len][len];

        for (int j = 0; j < len; j++) {
            for (int i = 0; i <= j; i++) {
                boolean judge = s.charAt(i) == s.charAt(j);
                dp[i][j] = (j - i > 2) ? dp[i + 1][j - 1] && judge : judge;

                if (dp[i][j] && j - i + 1 > max) {
                    max = j - i + 1;
                    ans = s.substring(i, j + 1);
                }
            }
        }
        return ans;
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
        int n = s.length();
        int ans = 0;

        if (n <= 0)
            return 0;

        boolean[][] dp = new boolean[n][n];

        // Base case: single letter substrings
        for (int i = 0; i < n; i++, ans++)
            dp[i][i] = true;

        // Base case: double letter substrings
        for (int i = 0; i < n - 1; i++) {
            dp[i][i + 1] = s.charAt(i) == s.charAt(i + 1);
            ans += dp[i][i + 1] ? 1 : 0;
        }

        // All other cases: substrings of length 3 to n
        for (int len = 3; len <= n; len++) {
            for (int i = 0, j = i + len - 1; j < n; i++, j++) {
                dp[i][j] = dp[i + 1][j - 1] && (s.charAt(i) == s.charAt(j));
                ans += dp[i][j] ? 1 : 0;
            }
        }
        return ans;
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

    // 9. Tree

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

    // 104. Maximum Depth of Binary Tree

    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;

        int leftHeight = maxDepth(root.left);
        int rightHeight = maxDepth(root.right);
        return 1 + Math.max(leftHeight, rightHeight);
    }

    // 100. Same Tree

    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null && q == null)
            return true;

        if (p != null && q != null)
            return p.val == q.val && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);

        return false;
    }

    // 226. Invert Binary Tree

    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;

        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
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

    int maxSum = Integer.MIN_VALUE;

    public int maxPathSum(TreeNode root) {
        maxPathSumDFS(root);
        return maxSum;
    }

    private int maxPathSumDFS(TreeNode node) {
        if (node == null) {
            return 0;
        }

        // max sum on the left and right sub-trees of node
        int leftGain = Math.max(maxPathSumDFS(node.left), 0);
        int rightGain = Math.max(maxPathSumDFS(node.right), 0);

        // the price to start a new path where `node` is a highest node
        int pathPrice = node.val + leftGain + rightGain;

        // update max_sum if it's better to start a new path
        maxSum = Math.max(maxSum, pathPrice);

        // for recursion :
        // return the max gain if continue the same path
        return node.val + Math.max(leftGain, rightGain);
    }

    // 102. Binary Tree Level Order Traversal

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

    // 572. Subtree of Another Tree

    public boolean isSubtree(TreeNode s, TreeNode t) {
        String spreorder = generatePreOrderString(s);
        String tpreorder = generatePreOrderString(t);
        return spreorder.contains(tpreorder);
    }

    private String generatePreOrderString(TreeNode node) {
        StringBuilder sb = new StringBuilder();
        Stack<TreeNode> stack = new Stack<>();
        stack.push(node);
        while (!stack.isEmpty()) {
            TreeNode current = stack.pop();
            if (current == null)
                sb.append(",#"); // Appending # inorder to handle same values but not subtree cases
            else {
                sb.append("," + current.val);
                stack.push(current.right);
                stack.push(current.left);
            }
        }
        return sb.toString();
    }

    // 105. Construct Binary Tree from Preorder and Inorder Traversal

    // Given two integer arrays preorder and inorder where preorder is the preorder
    // traversal of a binary tree and inorder is the inorder traversal of the same
    // tree, construct and return the binary tree.

    // Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
    // Output: [3,9,20,null,null,15,7]

    int preorderIndex;
    Map<Integer, Integer> inorderIndexMap;

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        preorderIndex = 0;
        // build a hashmap to store value -> its index relations
        inorderIndexMap = new HashMap<>();
        for (int i = 0; i < inorder.length; i++) {
            inorderIndexMap.put(inorder[i], i);
        }
        return arrayToTree(preorder, 0, preorder.length - 1);
    }

    private TreeNode arrayToTree(int[] preorder, int left, int right) {
        // if there are no elements to construct the tree
        if (left > right)
            return null;

        // select the preorder_index element as the root and increment it
        int rootValue = preorder[preorderIndex++];
        TreeNode root = new TreeNode(rootValue);

        // build left and right subtree
        // excluding inorderIndexMap[rootValue] element because it's the root
        root.left = arrayToTree(preorder, left, inorderIndexMap.get(rootValue) - 1);
        root.right = arrayToTree(preorder, inorderIndexMap.get(rootValue) + 1, right);

        return root;
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

    // 230. Kth Smallest Element in a BST

    // Given the root of a binary search tree, and an integer k, return the kth
    // smallest value (1-indexed) of all the values of the nodes in the tree.

    // Input: root = [5,3,6,2,4,null,null,1], k = 3
    // Output: 3

    public int kthSmallest(TreeNode root, int k) {
        List<Integer> nums = inorder(root, new ArrayList<Integer>());
        return nums.get(k - 1);
    }

    private ArrayList<Integer> inorder(TreeNode root, ArrayList<Integer> arr) {
        if (root == null)
            return arr;
        inorder(root.left, arr);
        arr.add(root.val);
        inorder(root.right, arr);
        return arr;
    }

    // 235. Lowest Common Ancestor of a Binary Search Tree

    // Given a binary search tree (BST), find the lowest common ancestor (LCA) of
    // two given nodes in the BST.

    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val) {
            return lowestCommonAncestor(root.left, p, q);
        } else if (p.val > root.val && q.val > root.val) {
            return lowestCommonAncestor(root.right, p, q);
        } else {
            return root;
        }
    }

    // 208. Implement Trie (Prefix Tree)

    // Implement the Trie class:

    // Trie() Initializes the trie object.
    // void insert(String word) Inserts the string word into the trie.
    // boolean search(String word) Returns true if the string word is in the trie
    // (i.e., was inserted before), and false otherwise.
    // boolean startsWith(String prefix) Returns true if there is a previously
    // inserted string word that has the prefix prefix, and false otherwise.

    class TrieNode {
        public char val;
        public boolean isWord;
        public TrieNode[] children = new TrieNode[26];

        public TrieNode() {
        }

        TrieNode(char c) {
            TrieNode node = new TrieNode();
            node.val = c;
        }
    }

    public class Trie {
        private TrieNode root;

        public Trie() {
            root = new TrieNode();
            root.val = ' ';
        }

        public void insert(String word) {
            TrieNode ws = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (ws.children[c - 'a'] == null) {
                    ws.children[c - 'a'] = new TrieNode(c);
                }
                ws = ws.children[c - 'a'];
            }
            ws.isWord = true;
        }

        public boolean search(String word) {
            TrieNode ws = root;
            for (int i = 0; i < word.length(); i++) {
                char c = word.charAt(i);
                if (ws.children[c - 'a'] == null)
                    return false;
                ws = ws.children[c - 'a'];
            }
            return ws.isWord;
        }

        public boolean startsWith(String prefix) {
            TrieNode ws = root;
            for (int i = 0; i < prefix.length(); i++) {
                char c = prefix.charAt(i);
                if (ws.children[c - 'a'] == null)
                    return false;
                ws = ws.children[c - 'a'];
            }
            return true;
        }
    }

    // 211. Design Add and Search Words Data Structure

    class TrieNode1 {
        Map<Character, TrieNode1> children = new HashMap<>();
        boolean word = false;

        public TrieNode1() {
        }
    }

    class WordDictionary {
        TrieNode1 trie;

        /** Initialize your data structure here. */
        public WordDictionary() {
            trie = new TrieNode1();
        }

        /** Adds a word into the data structure. */
        public void addWord(String word) {
            TrieNode1 node = trie;

            for (char ch : word.toCharArray()) {
                if (!node.children.containsKey(ch)) {
                    node.children.put(ch, new TrieNode1());
                }
                node = node.children.get(ch);
            }
            node.word = true;
        }

        /** Returns if the word is in the node. */
        public boolean searchInNode(String word, TrieNode1 node) {
            for (int i = 0; i < word.length(); ++i) {
                char ch = word.charAt(i);
                if (!node.children.containsKey(ch)) {
                    // if the current character is '.'
                    // check all possible nodes at this level
                    if (ch == '.') {
                        for (char x : node.children.keySet()) {
                            TrieNode1 child = node.children.get(x);
                            if (searchInNode(word.substring(i + 1), child)) {
                                return true;
                            }
                        }
                    }
                    // if no nodes lead to answer
                    // or the current character != '.'
                    return false;
                } else {
                    // if the character is found
                    // go down to the next level in trie
                    node = node.children.get(ch);
                }
            }
            return node.word;
        }

        /**
         * Returns if the word is in the data structure. A word could contain the dot
         * character '.' to represent any one letter.
         */
        public boolean search(String word) {
            return searchInNode(word, trie);
        }
    }

    // 212. Word Search II

    // Given an m x n board of characters and a list of strings words, return all
    // words on the board.

    // Each word must be constructed from letters of sequentially adjacent cells,
    // where adjacent cells are horizontally or vertically neighboring. The same
    // letter cell may not be used more than once in a word.

    // Example 1:
    // Input: board =
    // [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]],
    // words = ["oath","pea","eat","rain"]
    // Output: ["eat","oath"]

    class TrieNode2 {
        HashMap<Character, TrieNode2> children = new HashMap<>();
        String word = null;

        public TrieNode2() {
        }
    }

    char[][] _board = null;
    ArrayList<String> _result = new ArrayList<String>();

    public List<String> findWords(char[][] board, String[] words) {

        // Step 1). Construct the Trie
        TrieNode2 root = new TrieNode2();
        for (String word : words) {
            TrieNode2 node = root;

            for (Character letter : word.toCharArray()) {
                if (node.children.containsKey(letter)) {
                    node = node.children.get(letter);
                } else {
                    TrieNode2 newNode = new TrieNode2();
                    node.children.put(letter, newNode);
                    node = newNode;
                }
            }
            node.word = word; // store words in Trie
        }

        this._board = board;
        // Step 2). Backtracking starting for each cell in the board
        for (int row = 0; row < board.length; ++row) {
            for (int col = 0; col < board[row].length; ++col) {
                if (root.children.containsKey(board[row][col])) {
                    backtracking(row, col, root);
                }
            }
        }

        return this._result;
    }

    private void backtracking(int row, int col, TrieNode2 parent) {
        Character letter = this._board[row][col];
        TrieNode2 currNode = parent.children.get(letter);

        // check if there is any match
        if (currNode.word != null) {
            this._result.add(currNode.word);
            currNode.word = null;
        }

        // mark the current letter before the EXPLORATION
        this._board[row][col] = '#';

        // explore neighbor cells in around-clock directions: up, right, down, left
        int[] rowOffset = { -1, 0, 1, 0 };
        int[] colOffset = { 0, 1, 0, -1 };
        for (int i = 0; i < 4; ++i) {
            int newRow = row + rowOffset[i];
            int newCol = col + colOffset[i];
            if (newRow < 0 || newRow >= this._board.length || newCol < 0 || newCol >= this._board[0].length) {
                continue;
            }
            if (currNode.children.containsKey(this._board[newRow][newCol])) {
                backtracking(newRow, newCol, currNode);
            }
        }

        // End of EXPLORATION, restore the original letter in the board.
        this._board[row][col] = letter;

        // Optimization: incrementally remove the leaf nodes
        if (currNode.children.isEmpty()) {
            parent.children.remove(letter);
        }
    }

    // 10. Heap

    // 23. Merge k Sorted Lists

    // You are given an array of k linked-lists lists, each linked-list is sorted in
    // ascending order.

    // Merge all the linked-lists into one sorted linked-list and return it.

    // Example 1:

    // Input: lists = [[1,4,5],[1,3,4],[2,6]]
    // Output: [1,1,2,3,4,4,5,6]
    // Explanation: The linked-lists are:
    // [
    // 1->4->5,
    // 1->3->4,
    // 2->6
    // ]
    // merging them into one sorted list:
    // 1->1->2->3->4->4->5->6

    public ListNode mergeKLists1(ListNode[] lists) {
        // if(lists == null || lists.length == 0)
        // return new ListNode(null);

        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (ListNode listNode : lists) {
            while (listNode != null) {
                minHeap.add(listNode.val);
                listNode = listNode.next;
            }
        }

        ListNode last = new ListNode();
        ListNode first = last;
        while (!minHeap.isEmpty()) {
            last.next = new ListNode(minHeap.remove());
            last = last.next;
        }
        return first.next;
    }

    // 347. Top K Frequent Elements

    // Given an integer array nums and an integer k, return the k most frequent
    // elements. You may return the answer in any order.

    // Example 1:
    // Input: nums = [1,1,1,2,2,3], k = 2
    // Output: [1,2]

    // Example 2:
    // Input: nums = [1], k = 1
    // Output: [1]

    public int[] topKFrequent(int[] nums, int k) {
        if (k == nums.length)
            return nums;

        // 1. build hash map : character and how often it appears
        Map<Integer, Integer> count = new HashMap<>();
        for (int num : nums) {
            count.put(num, count.getOrDefault(num, 0) + 1);
        }

        // init heap 'the less frequent element first'
        Queue<Integer> heap = new PriorityQueue<>((n1, n2) -> count.get(n1) - count.get(n2));

        // 2. keep k top frequent elements in the heap
        for (int n : count.keySet()) {
            heap.add(n);
            if (heap.size() > k)
                heap.poll();
        }

        // 3. build an output array
        int[] top = new int[k];
        while (k-- > 0) {
            top[k] = heap.poll();
        }
        return top;
    }

    // 295. Find Median from Data Stream

    // The median is the middle value in an ordered integer list. If the size of the
    // list is even, there is no middle value and the median is the mean of the two
    // middle values.

    // For example, for arr = [2,3,4], the median is 3.
    // For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
    // Implement the MedianFinder class:

    // MedianFinder() initializes the MedianFinder object.
    // void addNum(int num) adds the integer num from the data stream to the data
    // structure.
    // double findMedian() returns the median of all elements so far.

    // Explanation
    // MedianFinder medianFinder = new MedianFinder();
    // medianFinder.addNum(1); // arr = [1]
    // medianFinder.addNum(2); // arr = [1, 2]
    // medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
    // medianFinder.addNum(3); // arr[1, 2, 3]
    // medianFinder.findMedian(); // return 2.0

    class MedianFinder {
        PriorityQueue<Integer> minH;
        PriorityQueue<Integer> maxH;

        /** initialize your data structure here. */
        public MedianFinder() {
            minH = new PriorityQueue<Integer>();
            maxH = new PriorityQueue<Integer>((ob1, ob2) -> ob2.compareTo(ob1));
        }

        public void addNum(int num) {
            maxH.add(num);
            minH.add(maxH.poll());
            if (minH.size() > maxH.size()) {
                maxH.add(minH.poll());
            }
        }

        public double findMedian() {
            if (minH.size() == maxH.size())
                return (double) (maxH.peek() + minH.peek()) * 0.5;
            else
                return (double) maxH.peek();
        }
    }

}
