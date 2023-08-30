package com.prem.problems;

import java.util.*;

public class LeetCode {

    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int val) {
            this.val = val;
        }
    }

    public String revereString(String str) {
        // two pointer approach
        char[] ch = str.toCharArray();
        int i = 0;
        int j = str.length() - 1;
        while (i < j) {
            char temp = ch[i];
            ch[i++] = ch[j];
            ch[j--] = temp;
        }
        return new String(ch);
    }

    public boolean isValidAnagram(String s, String t) {
        // s = anagram, t = nagaram, return true. rearrangement of characters. lower
        // case only
        if (s.length() != t.length())
            return false;
        int[] counts = new int[26];
        for (int i = 0; i < s.length(); i++) {
            counts[s.charAt(i) - 'a']++;
            counts[t.charAt(i) - 'a']--;
        }
        for (int count : counts) {
            if (count != 0)
                return false;
        }
        return true;
    }

    public int firstUniqueCharInString(String s) {
        // "leetcode". l is first unique, so return l's index
        Map<Character, Integer> map = new java.util.HashMap<>();
        for (int i = 0; i < s.length(); i++) {
            char current = s.charAt(i);
            if (!map.containsKey(current))
                map.put(current, i);
            else
                map.put(current, -1);
        }
        int min = Integer.MAX_VALUE;
        for (char c : map.keySet()) {
            if (map.get(c) > -1 && map.get(c) < min)
                min = map.get(c);
        }
        return min == Integer.MAX_VALUE ? -1 : min;
    }

    public int buySellStocks(int[] prices) {
        // Best time to buy and sell stocks [7, 1, 5, 3, 6, 4] -> 6 - 1 = 5, buy on day
        // 2 and sell on day 5
        int max = 0;
        int min = Integer.MAX_VALUE;
        for (int price : prices) {
            if (price < min)
                min = price;
            else
                max = Math.max(max, price - min);
        }
        return max;
    }

    public boolean containsDuplicate(int[] nums) {
        // [1,2,3,1]->true
        Set<Integer> set = new HashSet<>();
        for (int num : nums) {
            if (set.contains(num))
                return true;
            else
                set.add(num);
        }
        return false;
    }

    public List<String> fizzBuzz(int n) {
        List<String> result = new ArrayList<>();
        for (int i = 1; i <= n; i++) {
            if (i % 3 == 0 && i % 5 == 0)
                result.add("FizzBuzz");
            else if (i % 3 == 0)
                result.add("Fizz");
            else if (i % 5 == 0)
                result.add("Buzz");
            else
                result.add(Integer.toString(i));
        }
        return result;
    }

    public boolean validateParentheses(String s) {
        Stack<Character> stack = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(' || c == '{' || c == '[')
                stack.push(c);
            else if (c == ')' && !stack.isEmpty() && stack.peek() == '(')
                stack.pop();
            else if (c == '}' && !stack.isEmpty() && stack.peek() == '{')
                stack.pop();
            else if (c == ']' && !stack.isEmpty() && stack.peek() == '[')
                stack.pop();
            else
                return false;
        }
        return stack.isEmpty();
    }

    public int singleNumber(int[] nums) {
        // every list element appears twice except one -> find that number. [2,2,1]
        Set<Integer> set = new HashSet<>();
        for (int i : nums) {
            if (set.contains(i))
                set.remove(i);
            else
                set.add(i);
        }

        return set.iterator().next();
    }

    public boolean powerOfTwo(int n) {
        // input 1 -> 2 to the power of 0 is 1, so return true. 16 -> 2 to the power of
        // 4 is 16, so return true.
        long i = 1;
        while (i < n) {
            i *= 2;
        }
        return i == n;
    }

    public int numberOfIslands(char[][] grid) {
        if (grid == null || grid.length == 0)
            return 0;
        int result = 0;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == '1') {
                    result += islandsDFS(grid, i, j);
                }
            }
        }
        return result;
    }

    private int islandsDFS(char[][] grid, int i, int j) {
        if (i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == '0')
            return 0;

        grid[i][j] = '0';
        islandsDFS(grid, i + 1, j);
        islandsDFS(grid, i - 1, j);
        islandsDFS(grid, i, j + 1);
        islandsDFS(grid, i, j - 1);
        return 1;
    }

    public int findPeakElement(int[] nums) {
        // element that is greater than its neighbours
        // [1,2,3,1] -> 3's index 2
        // faster solution is binary search, array has to be sorted
        int left = 0;
        int right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[mid + 1])
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    public int[] twoSum(int[] nums, int target) {
        // return indices of two nums such that they add to a specific target [2, 7, 11,
        // 15] target = 9 -> return [0, 1]
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int diff = target - nums[i];
            if (map.containsKey(diff)) {
                result[0] = i;
                result[1] = map.get(diff);
                return result;
            }
            map.put(nums[i], i);
        }
        return result;
    }

    public boolean backSpaceStingCompare(String s, String t) {
        // ab#c, ad#c -> true, ab##, cd## -> true, ab#c, abc# -> false
        Stack<Character> sStack = new Stack<>();
        Stack<Character> tStack = new Stack<>();
        for (char ch : s.toCharArray()) {
            if (ch != '#') {
                sStack.push(ch);
            } else if (!sStack.empty()) {
                sStack.pop();
            }
        }
        for (char ch : t.toCharArray()) {
            if (ch != '#') {
                tStack.push(ch);
            } else if (!sStack.empty()) {
                tStack.pop();
            }
        }
        while (!sStack.isEmpty()) {
            char current = sStack.pop();
            if (!tStack.isEmpty() || tStack.pop() != current)
                return false;
        }
        return tStack.isEmpty();
    }

    public int[] moveZeros(int[] nums) {
        // move zeros to the end of the array [0,1,0,3,12] -> [1,3,12,0,0]
        // have an index and move the value to left once at the end then fill the rest
        // with zeros
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0)
                nums[index++] = nums[i];
        }
        for (int i = index; i < nums.length; i++) {
            nums[i] = 0;
        }
        return nums;
    }

    public int reverseInteger(int x) {
        // 32 bit signed int 123 -> 321, -123 -> -321
        boolean negative = false;
        if (x < 0) {
            negative = true;
            x *= -1;
        }
        long reversed = 0;
        while (x > 0) {
            reversed = (reversed * 10) + (x % 10);
            x /= 10;
        }
        if (reversed > Integer.MAX_VALUE)
            return -1;
        return negative ? (int) (reversed * -1) : (int) reversed;
    }

    public int[] plusOne(int[] digits) {
        // [4,3,2,1] -> [4,3,2,2]
        // [4,3,2,9] -> [4,3,3,0]
        for (int i = digits.length - 1; i >= 0; i--) {
            if (digits[i] < 9) {
                digits[i]++;
                return digits;
            } else {
                digits[i] = 0;
            }
        }
        int[] result = new int[digits.length + 1];
        result[0] = 1;
        return result;
    }

    public int firstBadVersion(int n) {
        // Given n = 5 and version = 4 is the first bad version
        // call isBadVersion(3) -> false
        // call isBadVersion(4) -> true
        // call isBadVersion(5) -> true
        // better solution is a binary search // 0001111111
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (!isBadVersion(mid))
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    private boolean isBadVersion(int x) {
        return false;
    }

    // 67. Add Binary
    // Given two binary strings a and b, return their sum as a binary string.

    // Example 1:

    // Input: a = "11", b = "1"
    // Output: "100"

    public String addBinary(String a, String b) {
        StringBuilder sb = new StringBuilder();
        int i = a.length() - 1;
        int j = b.length() - 1;
        int carry = 0;
        while (i >= 0 || j >= 0) {
            int sum = carry;
            if (i >= 0)
                sum += a.charAt(i--) - '0';
            if (j >= 0)
                sum += b.charAt(j--) - '0';
            sb.insert(0, sum % 2);
            carry = sum / 2;
        }
        if (carry > 0) {
            sb.insert(0, 1);
        }
        return sb.toString();
    }

    public int paintHouse(int[][] costs) {
        // [[17,2,17],
        //  [16,16,5],
        //  [14,3,19]] -> output 10.
        // paint house 0 into blue
        // paint house 1 into green
        // paint house 2 into blue
        // find the min cost to paint house
        if (costs == null || costs.length == 0)
            return 0;
        for (int i = 1; i < costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        return Math.min(Math.min(costs[costs.length - 1][0], costs[costs.length - 1][1]), costs[costs.length - 1][2]);
    }

    public boolean robotReturnToOrigin(String moves) {
        // UD -> true
        // LL -> false
        // judgeCircle
        int UD = 0;
        int LR = 0;
        for (int i = 0; i < moves.length(); i++) {
            char currentMove = moves.charAt(i);
            if (currentMove == 'U')
                UD++;
            else if (currentMove == 'D')
                UD--;
            else if (currentMove == 'L')
                LR++;
            else if (currentMove == 'R')
                LR--;
        }
        return UD == 0 && LR == 0;
    }

    public boolean containsDuplicate(int[] nums, int k) {
        // [1,2,3,1], k = 3 -> true
        // [1,2,3,1,2,3], k = 2 -> false
        // k is the difference of duplicates
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int current = nums[i];
            if (map.containsKey(current) && i - map.get(current) <= k)
                return true;
            else
                map.put(current, i);
        }
        return false;
    }

    public char findTheDifference(String s, String t) {
        // s=abcd, t=abcde, return e
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);
        for (char c : t.toCharArray()) {
            if ((map.containsKey(c) && map.get(c) == 0) || !map.containsKey(c))
                return c;
            else
                map.put(c, map.get(c) - 1);
        }
        return ' ';
    }

    public TreeNode convertSortedArrayToBST(int[] nums) {
        // [-10, -3, 0, 5, 9] -> [0, -3, 9, -10, null, 5]
        if (nums == null || nums.length == 0)
            return null;
        return constructBSTRecursive(nums, 0, nums.length - 1);
    }

    private TreeNode constructBSTRecursive(int[] nums, int left, int right) {
        if (left > right)
            return null;
        int mid = left + (right - left) / 2;
        TreeNode current = new TreeNode(nums[mid]);
        current.left = constructBSTRecursive(nums, left, mid - 1);
        current.right = constructBSTRecursive(nums, mid + 1, right);
        return current;

    }

    public int findTheCelebrity(int n) {
        // celeb -> all the n-1 ppl in the party know him/her but he/she don't know any
        // one of them
        int person = 0;
        for (int i = 1; i < n; i++) {
            if (knows(person, i)) {
                person = i;
            }
        }
        for (int i = 0; i < n; i++) {
            if (i != person && knows(person, i) || !knows(i, person))
                return -1;
        }
        return person;
    }

    private boolean knows(int a, int b) {
        // this helper will return true if either a knows b but b dont know a and vice
        // versa.
        return true;
    }

    public int[] intersectionOfTwoArrays(int[] nums1, int[] nums2) {
        // [1,2,2,1] [2,2] -> [2]
        Set<Integer> set = new HashSet<>();
        for (int num : nums1) {
            set.add(num);
        }
        Set<Integer> intersection = new HashSet<>();
        for (int num : nums2) {
            if (set.contains(num))
                intersection.add(num);
        }

        int[] result = new int[intersection.size()];
        int index = 0;
        for (int i : intersection)
            result[index++] = i;
        return result;
    }

    public int findTheMissingNumber(int[] nums) {
        // [3, 0 ,1] -> return 2
        // array {0, 1, 2,...n} n distinct elements
        // Gauss's rule/law [n(n-1)/2]
        int sum = 0;
        for (int i : nums)
            sum += i;
        int n = nums.length + 1; // includes 0
        return ((n * (n - 1)) / 2) - sum;
    }

    public boolean meetingRooms(Interval[] intervals) {
        // [[0,30][5,10][15,20]] -> false
        // [[7,10][2,4]] -> true
        // determine if a person could attend all meetings.
        int[] start = new int[intervals.length];
        int[] end = new int[intervals.length];
        for (int i = 0; i < intervals.length; i++) {
            start[i] = intervals[i].start;
            end[i] = intervals[i].end;
        }
        Arrays.sort(start);
        Arrays.sort(end);
        for (int i = 0; i < start.length; i++) {
            if (start[i + 1] < end[i])
                return false;
        }
        return true;
    }

    class Interval {
        int start;
        int end;

        Interval() {
            start = 0;
            end = 0;
        }

        Interval(int s, int e) {
            this.start = s;
            this.end = e;
        }
    }

    public int majorityElement(int[] nums) {
        // find the element that appears more than half the time
        // [3,2,3] -> 3
        if (nums.length == 1)
            return nums[0];
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums) {
            if (map.containsKey(i) && map.get(i) + 1 > nums.length / 2)
                return i;
            else
                map.put(i, map.getOrDefault(i, 0) + 1);
        }
        return -1;
    }

    public int hammingDistance(int x, int y) {
        // dist between two integers is the number of positions at which corresponding
        // bits are different.
        // x=1, y=4 -> 2
        // 1 (0 0 0 1)
        // 4 (0 1 0 0)
        // above rows the intersection bits are different at 2,4 columns (2 different
        // places the bits are not same. so, return 2)
        // XOR bit wise operation
        int count = 0;
        while (x > 0 || y > 0) {
            count += (x % 2) ^ (y % 2);
            x >>= 1; // this shifting just which is x/2
            y >>= 1;
        }
        return count;
    }

    public int bestTimeToBuyStocks(int[] prices) {
        // [7,1,5,3,6,4] -> buy on day 2(1) and sell on day 3(5) and buy on day 4(3) and
        // sell on day 5(6)
        if (prices == null || prices.length == 0)
            return 0;
        int profit = 0;
        for (int i = 0; i < prices.length; i++) {
            if (prices[i + 1] > prices[i])
                profit += prices[i + 1] - prices[i];
        }
        return profit;
    }

    // 137. Single Number II

    // Given an integer array nums where every element appears three times except
    // for one, which appears exactly once. Find the single element and return it.

    // You must implement a solution with a linear runtime complexity and use only
    // constant extra space.

    // Example 1:

    // Input: nums = [2,2,3,2]
    // Output: 3

    public int singleNumber2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i : nums)
            map.put(i, map.getOrDefault(i, 0) + 1);
        for (int key : map.keySet())
            if (map.get(key) == 1)
                return key;
        return -1;
    }

    public boolean pathSum(TreeNode root, int sum) {
        // given BST, sum the root to leaf such that equals to given sum
        if (root == null)
            return false;
        else if (root.left == null && root.right == null && sum - root.val == 0)
            return true;
        else
            return pathSum(root.left, sum - root.val) || pathSum(root.right, sum - root.val);
    }

    public int removeElement(int[] nums, int val) {
        int index = 0;
        for (int i : nums) {
            if (i != val)
                nums[index++] = i;
        }
        return index;
    }

    public int numberComplement(int num) {
        // flip the bits individually input 5(101) -> output 2(010)
        int result = 0;
        int power = 1;
        while (num > 0) {
            result += (num % 2 ^ 1) * power;
            power <<= 1; // multiply by 2
            num >>= 1; // divide by 2
        }
        return result;
    }

    public int climbingStairs(int n) {
        // how many ways we can climb the n stairs (at a time can step either 1 or 2
        // steps)
        // input = 2 -> (1+1, 2 steps) -> output = 2
        // input = 3 -> (1+1+1, 1+2, 2+1 steps) -> output = 3
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for (int i = 2; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }

    // 17. Letter Combinations of a Phone Number

    // Given a string containing digits from 2-9 inclusive, return all possible
    // letter combinations that the number could represent. Return the answer in any
    // order.

    // A mapping of digit to letters (just like on the telephone buttons) is given
    // below. Note that 1 does not map to any letters.

    // Example 1:

    // Input: digits = "23"
    // Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]

    public List<String> phoneNumberLetterCombinations(String digits) {
        // input -> 23-> [ad,ae, af, bd, be, bf, cd, ce, cf]
        List<String> result = new ArrayList<>();
        if (digits == null || digits.length() == 0)
            return result;
        String[] mappings = { "0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz" };
        combo(result, digits, "", 0, mappings);
        return result;
    }

    private void combo(List<String> result, String digits, String current, int index, String[] mappings) {
        if (index == digits.length()) {
            result.add(current);
            return;
        }
        String letters = mappings[digits.charAt(index) - '0'];
        for (int i = 0; i < letters.length(); i++) {
            combo(result, digits, current + letters.charAt(i), index + 1, mappings);
        }
    }

    public TreeNode lowestCommonAncestorInBST(TreeNode root, TreeNode p, TreeNode q) {
        if (p.val < root.val && q.val < root.val)
            return lowestCommonAncestorInBST(root.left, p, q);
        else if (p.val > root.val && q.val > root.val)
            return lowestCommonAncestorInBST(root.right, p, q);
        else
            return root;
    }

    public int jewelsStones(String j, String s) {
        Set<Character> set = new HashSet<>();
        for (char c : j.toCharArray())
            set.add(c);
        int numJewel = 0;
        for (char c : s.toCharArray())
            if (set.contains(c))
                numJewel++;
        return numJewel;
    }

    public boolean binaryNumberWithAlternatingBits(int n) {
        // 5 -> 101 true, 7 -> 111 false, 2 -> 010 true, 10 -> 1010 true, 11 -> 1011 ->
        // false
        int last = n % 2;
        n >>= 1; // diving by 2
        while (n > 0) {
            int current = n % 2;
            if (current == last)
                return false;
            last = current;
            n >>= 1;
        }
        return true;
    }

    // 125. Valid Palindrome

    public boolean validatePalindrome(String s) {
        // race car
        int start = 0;
        int end = s.length() - 1;
        while (start < end) {
            while (start < end && !Character.isLetterOrDigit(s.charAt(start)))
                start++;
            while (start < end && !Character.isLetterOrDigit(s.charAt(end)))
                end--;
            if (start < end && Character.toLowerCase(s.charAt(start++)) != Character.toLowerCase(s.charAt(end--)))
                return false;
        }
        return true;
    }

    // 11. Container With Most Water

    public int containerWithMostWater(int[] height) {
        /*
         * int max = Integer.MIN_VALUE; for(int i=0;i<height.length;i++){ for(int j=i+1;
         * j<height.length;j++){ int min = Math.min(height[i], height[j]); //
         * restriction to overflow the water max = Math.max(max, min * (j - i)); } }
         * return max;
         */

        int max = Integer.MIN_VALUE;
        int i = 0;
        int j = height.length - 1;
        while (i < j) {
            int min = Math.min(height[i], height[j]); // restriction to overflow the water
            max = Math.max(max, min * (j - i));
            if (height[i] < height[j])
                i++;
            else
                j--;
        }
        return max;
    }

    // 896. Monotonic Array

    // An array is monotonic if it is either monotone increasing or monotone
    // decreasing.

    // Example 2:

    // Input: nums = [6,5,4,4]
    // Output: true
    // Example 3:

    // Input: nums = [1,3,2]
    // Output: false

    public boolean isMonotonic(int[] A) {
        boolean inc = true;
        boolean dec = true;

        for (int i = 0; i < A.length - 1; i++) {
            if (A[i] > A[i + 1])
                inc = false;
            if (A[i] < A[i + 1])
                dec = false;
        }
        return inc || dec;
    }

    // 404. Sum of Left Leaves

    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null)
            return 0;
        else if (root.left != null && root.left.left == null && root.left.right == null)
            return root.left.val + sumOfLeftLeaves(root.right);
        else
            return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);

    }

    // 26. Remove Duplicates from Sorted Array

    // Example 1:

    // Input: nums = [1,1,2]
    // Output: 2, nums = [1,2,_]
    // Explanation: Your function should return k = 2, with the first two elements
    // of nums being 1 and 2 respectively.
    // It does not matter what you leave beyond the returned k (hence they are
    // underscores).
    // Example 2:

    // Input: nums = [0,0,1,1,1,2,2,3,3,4]
    // Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
    // Explanation: Your function should return k = 5, with the first five elements
    // of nums being 0, 1, 2, 3, and 4 respectively.
    // It does not matter what you leave beyond the returned k (hence they are
    // underscores).

    public int removeDuplicates(int[] nums) {
        int index = 1;
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] != nums[i + 1]) {
                nums[index++] = nums[i + 1];
            }
        }

        return index;
    }

    public List<Integer> bstPostOrderTraversal(TreeNode root) {
        List<Integer> values = new ArrayList<>();
        if (root == null)
            return values;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while (!stack.isEmpty()) {
            TreeNode current = stack.pop();
            values.add(0, current.val); // always push the existing values and add to the 1st position
            if (current.left != null)
                stack.push(current.left);
            if (current.right != null)
                stack.push(current.right);
        }
        return values;
    }

    public int battleShipsInBoard(char[][] board) {
        int numBattleShips = 0;
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                // Optimized
                if (board[i][j] == '.')
                    continue;
                if (i > 0 && board[i - 1][j] == 'X')
                    continue;
                if (j > 0 && board[i][j - 1] == 'X')
                    continue;
                numBattleShips++;
            }
        }
        return numBattleShips;
    }

    private void sinkShips(char[][] board, int i, int j) {
        if (i < 0 || i >= board.length || j < 0 || j > board[i].length || board[i][j] != 'X')
            return;

        board[i][j] = '.';
        sinkShips(board, i + 1, j);
        sinkShips(board, i - 1, j);
        sinkShips(board, i, j + 1);
        sinkShips(board, i, j - 1);

    }

    public int[][] flippingImage(int[][] A) {
        for (int i = 0; i < A.length; i++) {
            int j = 0;
            int k = A[i].length - 1;
            while (j < k) {
                int temp = A[i][j];
                A[i][j++] = A[i][k];
                A[i][k--] = temp;
            }

            for (j = 0; j < A[i].length; j++)
                A[i][j] = A[i][j] == 1 ? 0 : 1;
        }
        return A;
    }

    public int kthLargestElementInArray(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int i : nums) {
            minHeap.add(i);
            if (minHeap.size() > k)
                minHeap.remove();
        }
        return minHeap.remove();
    }

    public int totalFruit(int[] tree) {
        // longest substring with k(2) characters
        if (tree == null || tree.length == 0)
            return 0;
        int max = 1;
        Map<Integer, Integer> map = new HashMap<>();
        int i = 0;
        int j = 0;
        while (j < tree.length) {
            if (map.size() <= 2) {
                map.put(tree[j], j++);
            }
            if (map.size() > 2) {
                int min = tree.length - 1;
                for (int value : map.values()) {
                    min = Math.min(min, value);
                }
                i = min + 1;
                map.remove(tree[min]);
            }
            max = Math.max(max, j - i);
        }
        return max;
    }

    // 91. Decode Ways

    // decode an encoded message, all the digits must be grouped then mapped back
    // into letters using the reverse of the mapping above (there may be multiple
    // ways). For example, "11106" can be mapped into:

    // "AAJF" with the grouping (1 1 10 6)
    // "KJF" with the grouping (11 10 6)
    // Note that the grouping (1 11 06) is invalid because "06" cannot be mapped
    // into 'F' since "6" is different from "06".

    public int numDecodings(String s) {
        // 'A' -> 1
        // 'B' -> 2
        // 'Z' -> 26
        // '12' -> AB or L -> 2 occurrence
        // '226' -> "BZ" (2 26), "VF" (22 6), or "BBF" (2 2 6) -> 3 occurences
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i = 2; i <= s.length(); i++) {
            int oneDigit = Integer.valueOf(s.substring(i - 1, i));
            int twoDigit = Integer.valueOf(s.substring(i - 2, i));
            if (oneDigit >= 1) {
                dp[i] += dp[i - 1];
            }
            if (twoDigit >= 10 && twoDigit <= 26) {
                dp[i] += dp[i - 2];
            }
        }
        return dp[s.length()];
    }

    public List<List<Integer>> binaryTreeLevelOrderTraversal(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if (root == null)
            return result;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> currentLevel = new ArrayList<>();
            for (int i = 0; i < size; i++) {
                TreeNode current = q.remove();
                currentLevel.add(current.val);
                if (current.left != null)
                    q.add(current.left);
                if (current.right != null)
                    q.add(current.right);
            }
            result.add(currentLevel);
        }
        return result;
    }

    public int meetingRooms2(Interval[] intervals) {
        if (intervals == null || intervals.length == 0)
            return 0;
        Arrays.sort(intervals, (a, b) -> a.start - b.start);
        PriorityQueue<Interval> minHeap = new PriorityQueue<>((a, b) -> a.end - b.end);
        minHeap.add(intervals[0]);
        for (int i = 1; i < intervals.length; i++) {
            Interval current = intervals[i];
            Interval earliest = minHeap.remove();
            if (current.start >= earliest.end)
                earliest.end = current.end;
            else
                minHeap.add(current);
            minHeap.add(earliest);
        }
        return minHeap.size();
    }

    public int[][] kClosestPoint(int[][] points, int k) {
        // input [[1,3][-2,2]] k = 1 -> output [[-2,2]]
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                (a, b) -> (b[0] * b[0] + b[1] * b[1]) - (a[0] * a[0] + a[1] * a[1]));
        // d = sqrt((x2-x1)+(y2-y1))
        for (int[] point : points) {
            maxHeap.add(point);
            if (maxHeap.size() > k)
                maxHeap.remove();
        }
        int[][] result = new int[k][2];
        while (k-- > 0) {
            result[k] = maxHeap.remove();
        }
        return result;
    }

    public String sortCharactersByFrequency(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : s.toCharArray())
            map.put(c, map.getOrDefault(c, 0) + 1);
        PriorityQueue<Character> maxHeap = new PriorityQueue<>((a, b) -> map.get(b) - map.get(a));
        maxHeap.addAll(map.keySet());
        StringBuilder sb = new StringBuilder();
        while (!maxHeap.isEmpty()) {
            char current = (char) maxHeap.remove();
            for (int i = 0; i < map.get(current); i++) {
                sb.append(current);
            }
        }
        return sb.toString();
    }

    public boolean isAlienSorted(String[] words, String order) {

        // Example 1:
        // Input: words = ["hello","leetcode"], order = "hlabcdefgijkmnopqrstuvwxyz"
        // Output: true
        // Explanation: As 'h' comes before 'l' in this language, then the sequence is
        // sorted.
        // Example 2:

        // Input: words = ["word","world","row"], order = "worldabcefghijkmnpqstuvxyz"
        // Output: false
        // Explanation: As 'd' comes after 'l' in this language, then words[0] >
        // words[1], hence the sequence is unsorted.
        // Example 3:

        // Input: words = ["apple","app"], order = "abcdefghijklmnopqrstuvwxyz"
        // Output: false
        // Explanation: The first three characters "app" match, and the second string is
        // shorter (in size.) According to lexicographical rules "apple" > "app",
        // because 'l' > '∅', where '∅' is defined as the blank character which is less
        // than any other character

        int[] alphabets = new int[26];
        for (int i = 0; i < order.length(); i++) {
            alphabets[order.charAt(i) - 'a'] = i;
        }

        for (int i = 0; i < words.length; i++) {
            for (int j = i + 1; j < words.length; j++) {
                int min = Math.min(words[i].length(), words[j].length());
                for (int k = 0; k < min; k++) {
                    char iChar = words[i].charAt(k);
                    char jChar = words[j].charAt(k);
                    if (alphabets[iChar - 'a'] < alphabets[jChar - 'a'])
                        break;
                    else if (alphabets[iChar - 'a'] > alphabets[jChar - 'a'])
                        return false;
                    else if (k == min - 1 && words[i].length() > words[j].length()) {
                        return false;
                    }
                }
            }
        }
        return true;
    }

    public String minRemoveToMakeValid(String s) {

        // Example 1:
        // Input: s = "lee(t(c)o)de)"
        // Output: "lee(t(c)o)de"
        // Explanation: "lee(t(co)de)" , "lee(t(c)ode)" would also be accepted.
        // Example 2:

        // Input: s = "a)b(c)d"
        // Output: "ab(c)d"
        // Example 3:

        // Input: s = "))(("
        // Output: ""
        // Explanation: An empty string is also valid.
        // Example 4:

        // Input: s = "(a(b(c)d)"
        // Output: "a(b(c)d)"

        StringBuilder sb = new StringBuilder();
        int balance = 0;
        for (char ch : s.toCharArray()) {
            if (ch == '(')
                balance++;
            else if (ch == ')') {
                if (balance == 0)
                    continue;
                balance--;
            }
            sb.append(ch);
        }
        StringBuilder result = new StringBuilder();
        for (int i = sb.length() - 1; i >= 0; i--) {
            if (sb.charAt(i) == '(' && balance-- > 0)
                continue;
            result.append(sb.charAt(i));
        }
        return result.reverse().toString();
    }

    // 1428. Leftmost Column with at Least a One
    // Linear Serch approach
    public int leftMostColumnWithOne1(BinaryMatrix binaryMatrix) {
        int rows = binaryMatrix.dimensions().get(0);
        int cols = binaryMatrix.dimensions().get(1);
        int smallestIndex = cols;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (binaryMatrix.get(row, col) == 1) {
                    smallestIndex = Math.min(smallestIndex, col);
                }
            }
        }
        return smallestIndex == cols ? -1 : smallestIndex;
    }

    // Binary Search Approach
    public int leftMostColumnWithOne2(BinaryMatrix binaryMatrix) {
        int rows = binaryMatrix.dimensions().get(0);
        int cols = binaryMatrix.dimensions().get(1);
        int smallestIndex = cols;
        for (int row = 0; row < rows; row++) {
            int lo = 0;
            int hi = cols - 1;
            while (lo < hi) {
                int mid = (lo + hi) / 2;
                if (binaryMatrix.get(row, mid) == 0)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (binaryMatrix.get(row, lo) == 1) {
                smallestIndex = Math.min(smallestIndex, lo);
            }
        }
        return smallestIndex == cols ? -1 : smallestIndex;
    }

    class BinaryMatrix {

        // Sample
        public List<Integer> dimensions() {
            return Arrays.asList();
        }

        // Sample
        public int get(int row, int col) {
            return 0;
        }
    }

    // 680. Valid Palindrome II
    // Given a string s, return true if the s can be palindrome after deleting at
    // most one character from it.
    // Input: s = "aba"
    // Output: true
    // Input: s = "abca"
    // Output: true
    // Input: s = "abc"
    // Output: false
    public boolean validPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        for (; i < j; i++, j--) {
            if (s.charAt(i) != s.charAt(j)) {
                int i1 = i;
                int i2 = i + 1;
                int j1 = j - 1;
                int j2 = j;
                while (i1 < j1 && s.charAt(i1) == s.charAt(j1)) {
                    i1++;
                    j1--;
                }
                while (i2 < j2 && s.charAt(i2) == s.charAt(j2)) {
                    i2++;
                    j2--;
                }
                return i1 >= j1 || i2 >= j2;
            }
        }
        return i >= j;
    }

    // 973. K Closest Points to Origin
    // Given an array of points where points[i] = [xi, yi] represents a point on the
    // X-Y plane and an integer k, return the k closest points to the origin (0, 0).
    // The distance between two points on the X-Y plane is the Euclidean distance
    // (i.e., √(x1 - x2)2 + (y1 - y2)2).
    // You may return the answer in any order. The answer is guaranteed to be unique
    // (except for the order that it is in).
    // Input: points = [[1,3],[-2,2]], k = 1
    // Output: [[-2,2]]
    // Explanation:
    // The distance between (1, 3) and the origin is sqrt(10).
    // The distance between (-2, 2) and the origin is sqrt(8).
    // Since sqrt(8) < sqrt(10), (-2, 2) is closer to the origin.
    // We only want the closest k = 1 points from the origin, so the answer is just
    // [[-2,2]].

    // Input: points = [[3,3],[5,-1],[-2,4]], k = 2
    // Output: [[3,3],[-2,4]]
    // Explanation: The answer [[-2,4],[3,3]] would also be accepted.

    public int[][] kClosest(int[][] points, int K) {
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>(
                (x, y) -> (y[0] * y[0] + y[1] * y[1] - x[0] * x[0] + x[1] * x[1]));
        for (int[] point : points) {
            maxHeap.add(point);
            if (maxHeap.size() > K) {
                maxHeap.remove();
            }
        }
        int[][] result = new int[K][2];
        while (K-- > 0) {
            result[K] = maxHeap.remove();
        }
        return result;
    }

    // 415. Add Strings
    // Given two non-negative integers, num1 and num2 represented as string, return
    // the sum of num1 and num2 as a string.
    // You must solve the problem without using any built-in library for handling
    // large integers (such as BigInteger). You must also not convert the inputs to
    // integers directly.
    // Input: num1 = "11", num2 = "123"
    // Output: "134"
    // Input: num1 = "456", num2 = "77"
    // Output: "533"

    public String addStrings(String num1, String num2) {
        int carry = 0;
        StringBuilder res = new StringBuilder();
        int p1 = num1.length() - 1;
        int p2 = num2.length() - 1;
        while (p1 >= 0 || p2 >= 0) {
            int x1 = p1 >= 0 ? num1.charAt(p1) - '0' : 0;
            int x2 = p2 >= 0 ? num2.charAt(p2) - '0' : 0;
            int sum = (x1 + x2 + carry) % 10;
            carry = (x1 + x2 + carry) / 10;
            res.append(sum);
            p1--;
            p2--;
        }
        if (carry != 0) {
            res.append(carry);
        }
        return res.reverse().toString();
    }

    // 560. Subarray Sum Equals K
    // Given an array of integers nums and an integer k, return the total number of
    // continuous subarrays whose sum equals to k.
    // Input: nums = [1,1,1], k = 2
    // Output: 2
    // Input: nums = [1,2,3], k = 3
    // Output: 2
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        for (int i = 0; i < nums.length; i++) {
            int sum = 0;
            for (int j = i; j < nums.length; j++) {
                sum += nums[j];
                if (sum == k)
                    count++;
            }
        }
        return count;
    }

    // 238. Product of Array Except Self
    // Given an integer array nums, return an array answer such that answer[i] is
    // equal to the product of all the elements of nums except nums[i].
    // Input: nums = [1,2,3,4]
    // Output: [24,12,8,6]
    // Input: nums = [-1,1,0,-3,3]
    // Output: [0,0,9,0,0]

    // 56. Merge Intervals
    // Given an array of intervals where intervals[i] = [starti, endi], merge all
    // overlapping intervals, and return an array of the non-overlapping intervals
    // that cover all the intervals in the input.
    // Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
    // Output: [[1,6],[8,10],[15,18]]
    // Explanation: Since intervals [1,3] and [2,6] overlaps, merge them into [1,6].
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

    // 301. Remove Invalid Parentheses
    // Given a string s that contains parentheses and letters, remove the minimum
    // number of invalid parentheses to make the input string valid.
    // Return all the possible results. You may return the answer in any order.

    // Example 1:
    // Input: s = "()())()"
    // Output: ["(())()","()()()"]

    // Example 2:
    // Input: s = "(a)())()"
    // Output: ["(a())()","(a)()()"]

    // Example 3:
    // Input: s = ")("
    // Output: [""]

    private Set<String> validExpressions = new HashSet<String>();

    private void recurse(String s, int index, int leftCount, int rightCount, int leftRem, int rightRem,
            StringBuilder expression) {
        // If we reached the end of the string, just check if the resulting expression
        // is
        // valid or not and also if we have removed the total number of left and right
        // parentheses that we should have removed.
        if (index == s.length()) {
            if (leftRem == 0 && rightRem == 0) {
                this.validExpressions.add(expression.toString());
            }
        } else {
            char character = s.charAt(index);
            int length = expression.length();

            // The discard case. Note that here we have our pruning condition.
            // We don't recurse if the remaining count for that parenthesis is == 0.
            if ((character == '(' && leftRem > 0) || (character == ')' && rightRem > 0)) {
                this.recurse(s, index + 1, leftCount, rightCount, leftRem - (character == '(' ? 1 : 0),
                        rightRem - (character == ')' ? 1 : 0), expression);
            }
            expression.append(character);

            // Simply recurse one step further if the current character is not a
            // parenthesis.
            if (character != '(' && character != ')') {
                this.recurse(s, index + 1, leftCount, rightCount, leftRem, rightRem, expression);
            } else if (character == '(') {
                // Consider an opening bracket.
                this.recurse(s, index + 1, leftCount + 1, rightCount, leftRem, rightRem, expression);
            } else if (rightCount < leftCount) {
                // Consider a closing bracket.
                this.recurse(s, index + 1, leftCount, rightCount + 1, leftRem, rightRem, expression);
            }
            // Delete for backtracking.
            expression.deleteCharAt(length);
        }
    }

    public List<String> removeInvalidParentheses(String s) {

        int left = 0, right = 0;
        // First, we find out the number of misplaced left and right parentheses.
        for (int i = 0; i < s.length(); i++) {
            // Simply record the left one.
            if (s.charAt(i) == '(') {
                left++;
            } else if (s.charAt(i) == ')') {
                // If we don't have a matching left, then this is a misplaced right, record it.
                right = left == 0 ? right + 1 : right;
                // Decrement count of left parentheses because we have found a right
                // which CAN be a matching one for a left.
                left = left > 0 ? left - 1 : left;
            }
        }
        this.recurse(s, 0, 0, 0, left, right, new StringBuilder());
        return new ArrayList<String>(this.validExpressions);
    }

    // 273. Integer to English Words
    // Convert a non-negative integer num to its English words representation.
    // Example 1:
    // Input: num = 123
    // Output: "One Hundred Twenty Three"
    // Example 2:
    // Input: num = 12345
    // Output: "Twelve Thousand Three Hundred Forty Five"
    // Example 3:
    // Input: num = 1234567
    // Output: "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty
    // Seven"
    // Example 4:
    // Input: num = 1234567891
    // Output: "One Billion Two Hundred Thirty Four Million Five Hundred Sixty Seven
    // Thousand Eight Hundred Ninety One"

    // 1234567890 -> 1.234.567.890 -> 1 Billion 234 Million 567 Thousand 890 and
    // reduces the initial problem to how to convert 3-digit integer to English
    // word. One could split further 234 -> 2 Hundred 34 into two sub-problems :
    // convert 1-digit integer and convert 2-digit integer. The first one is
    // trivial. The second one could be reduced to the first one for all 2-digit
    // integers but the ones from 10 to 19 which should be considered separately.

    public String numberToWords(int num) {
        if (num == 0)
            return "Zero";

        int billion = num / 1000000000;
        int million = (num - billion * 1000000000) / 1000000;
        int thousand = (num - billion * 1000000000 - million * 1000000) / 1000;
        int rest = num - billion * 1000000000 - million * 1000000 - thousand * 1000;

        String result = "";
        if (billion != 0)
            result = three(billion) + " Billion";
        if (million != 0) {
            if (!result.isEmpty())
                result += " ";
            result += three(million) + " Million";
        }
        if (thousand != 0) {
            if (!result.isEmpty())
                result += " ";
            result += three(thousand) + " Thousand";
        }
        if (rest != 0) {
            if (!result.isEmpty())
                result += " ";
            result += three(rest);
        }
        return result;
    }

    public String one(int num) {
        switch (num) {
            case 1:
                return "One";
            case 2:
                return "Two";
            case 3:
                return "Three";
            case 4:
                return "Four";
            case 5:
                return "Five";
            case 6:
                return "Six";
            case 7:
                return "Seven";
            case 8:
                return "Eight";
            case 9:
                return "Nine";
        }
        return "";
    }

    public String twoLessThan20(int num) {
        switch (num) {
            case 10:
                return "Ten";
            case 11:
                return "Eleven";
            case 12:
                return "Twelve";
            case 13:
                return "Thirteen";
            case 14:
                return "Fourteen";
            case 15:
                return "Fifteen";
            case 16:
                return "Sixteen";
            case 17:
                return "Seventeen";
            case 18:
                return "Eighteen";
            case 19:
                return "Nineteen";
        }
        return "";
    }

    public String ten(int num) {
        switch (num) {
            case 2:
                return "Twenty";
            case 3:
                return "Thirty";
            case 4:
                return "Forty";
            case 5:
                return "Fifty";
            case 6:
                return "Sixty";
            case 7:
                return "Seventy";
            case 8:
                return "Eighty";
            case 9:
                return "Ninety";
        }
        return "";
    }

    public String two(int num) {
        if (num == 0)
            return "";
        else if (num < 10)
            return one(num);
        else if (num < 20)
            return twoLessThan20(num);
        else {
            int tenner = num / 10;
            int rest = num - tenner * 10;
            if (rest != 0)
                return ten(tenner) + " " + one(rest);
            else
                return ten(tenner);
        }
    }

    public String three(int num) {
        int hundred = num / 100;
        int rest = num - hundred * 100;
        String res = "";
        if (hundred * rest != 0)
            res = one(hundred) + " Hundred " + two(rest);
        else if ((hundred == 0) && (rest != 0))
            res = two(rest);
        else if ((hundred != 0) && (rest == 0))
            res = one(hundred) + " Hundred";
        return res;
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

    // 938. Range Sum of BST
    // Given the root node of a binary search tree and two integers low and high,
    // return the sum of values of all nodes with a value in the inclusive range
    // [low, high].

    int ans;

    public int rangeSumBST(TreeNode root, int L, int R) {
        ans = 0;
        dfs(root, L, R);
        return ans;
    }

    public void dfs(TreeNode node, int L, int R) {
        if (node != null) {
            if (L <= node.val && node.val <= R)
                ans += node.val;
            if (L < node.val)
                dfs(node.left, L, R);
            if (node.val < R)
                dfs(node.right, L, R);
        }
    }

    // 426. Convert Binary Search Tree to Sorted Doubly Linked List
    // Convert a Binary Search Tree to a sorted Circular Doubly-Linked List in
    // place.
    class Node {
        public int val;
        public Node left;
        public Node right;

        public Node() {
        }

        public Node(int _val) {
            val = _val;
        }

        public Node(int _val, Node _left, Node _right) {
            val = _val;
            left = _left;
            right = _right;
        }
    };

    Node first = null;
    Node last = null;

    public Node treeToDoublyList(Node root) {
        if (root == null)
            return null;
        treeToDoublyListRec(root);
        last.right = first;
        first.left = last;
        return first;
    }

    private void treeToDoublyListRec(Node node) {
        if (node != null) {
            treeToDoublyListRec(node.left);
            if (last != null) {
                last.right = node;
                node.left = last;
            } else {
                first = node;
            }
            last = node;
            treeToDoublyListRec(node.right);
        }
    }

    // 523. Continuous Subarray Sum
    // Given an integer array nums and an integer k, return true if nums has a
    // continuous subarray of size at least two whose elements sum up to a multiple
    // of k, or false otherwise.
    // An integer x is a multiple of k if there exists an integer n such that x = n
    // * k. 0 is always a multiple of k.

    // Example 1:
    // Input: nums = [23,2,4,6,7], k = 6
    // Output: true
    // Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up
    // to 6.
    // Example 2:
    // Input: nums = [23,2,6,4,7], k = 6
    // Output: true
    // Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose
    // elements sum up to 42.
    // 42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
    // Example 3:
    // Input: nums = [23,2,6,4,7], k = 13
    // Output: false

    // We iterate through the input array exactly once, keeping track of the running
    // sum mod k of the elements in the process. If we find that a running sum value
    // at index j has been previously seen before in some earlier index i in the
    // array, then we know that the sub-array (i,j] contains a desired sum.

    public boolean checkSubarraySum(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>() {
            {
                put(0, -1);
            }
        };
        int runningSum = 0;
        for (int i = 0; i < nums.length; i++) {
            runningSum += nums[i];
            if (k != 0)
                runningSum %= k;
            Integer prev = map.get(runningSum);
            if (prev != null) {
                if (i - prev > 1)
                    return true;
            } else
                map.put(runningSum, i);
        }
        return false;
    }

    // 543. Diameter of Binary Tree
    // Given the root of a binary tree, return the length of the diameter of the
    // tree.
    // The diameter of a binary tree is the length of the longest path between any
    // two nodes in a tree. This path may or may not pass through the root.
    // The length of a path between two nodes is represented by the number of edges
    // between them.
    // Input: root = [1,2,3,4,5]
    // Output: 3
    // Explanation: 3 is the length of the path [4,2,1,3] or [5,2,1,3].
    // 1
    // / \
    // 2 3
    // / \
    // 4 5

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
        return Math.max(leftPath, rightPath) + 1;
    }

    // 621. Task Scheduler
    // Given a characters array tasks, representing the tasks a CPU needs to do,
    // where each letter represents a different task. Tasks could be done in any
    // order. Each task is done in one unit of time. For each unit of time, the CPU
    // could complete either one task or just be idle.
    // However, there is a non-negative integer n that represents the cooldown
    // period between two same tasks (the same letter in the array), that is that
    // there must be at least n units of time between any two same tasks.
    // Return the least number of units of times that the CPU will take to finish
    // all the given tasks.
    // Input: tasks = ["A","A","A","B","B","B"], n = 2
    // Output: 8
    // Explanation:
    // A -> B -> idle -> A -> B -> idle -> A -> B
    // There is at least 2 units of time between any two same tasks.

    // Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
    // Output: 16
    // Explanation:
    // One possible solution is
    // A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle
    // -> idle -> A

    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> map = new HashMap<>();
        for (char c : tasks) {
            map.put(c, map.getOrDefault(c, 0) + 1);
        }

        PriorityQueue<Integer> maxHeap = new PriorityQueue<>((a, b) -> b - a);
        maxHeap.addAll(map.values());

        int cycles = 0;
        while (!maxHeap.isEmpty()) {
            List<Integer> list = new ArrayList<>();
            for (int i = 0; i < n + 1; i++) {
                if (!maxHeap.isEmpty())
                    list.add(maxHeap.remove());
            }
            for (int i : list) {
                if (--i > 0)
                    maxHeap.add(i);
            }
            cycles += maxHeap.isEmpty() ? list.size() : n + 1;
        }
        return cycles;
    }

    // 987. Vertical Order Traversal of a Binary Tree
    // Given the root of a binary tree, calculate the vertical order traversal of
    // the binary tree.
    // For each node at position (row, col), its left and right children will be at
    // positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of
    // the tree is at (0, 0).
    // The vertical order traversal of a binary tree is a list of top-to-bottom
    // orderings for each column index starting from the leftmost column and ending
    // on the rightmost column. There may be multiple nodes in the same row and same
    // column. In such a case, sort these nodes by their values.
    // Return the vertical order traversal of the binary tree.

    // // Time -> O(n*log(n))
    // Space -> O(N) (priority queue)
    Queue<int[]> nodeEntries;

    public List<List<Integer>> verticalTraversal(TreeNode root) {
        nodeEntries = new PriorityQueue<>((e1, e2) -> {
            // 0 -> col, 1 -> row, 2 -> value
            for (int i = 0; i < e1.length; ++i) {
                if (e1[i] != e2[i])
                    return e1[i] - e2[i];
            }
            return 0;
        });
        dfsVerticalTraversal(root, 0, 0);
        List<List<Integer>> output = new ArrayList<>();
        int currentCol = Integer.MIN_VALUE;
        while (!nodeEntries.isEmpty()) {
            int[] entry = nodeEntries.remove();
            if (entry[0] != currentCol) {
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

    // 1762. Buildings With an Ocean View
    // There are n buildings in a line. You are given an integer array heights of
    // size n that represents the heights of the buildings in the line.
    // The ocean is to the right of the buildings. A building has an ocean view if
    // the building can see the ocean without obstructions. Formally, a building has
    // an ocean view if all the buildings to its right have a smaller height.
    // Return a list of indices (0-indexed) of buildings that have an ocean view,
    // sorted in increasing order.
    // Example 1:

    // Input: heights = [4,2,3,1]
    // Output: [0,2,3]
    // Explanation: Building 1 (0-indexed) does not have an ocean view because
    // building 2 is taller.
    // Example 2:

    // Input: heights = [4,3,2,1]
    // Output: [0,1,2,3]
    // Explanation: All the buildings have an ocean view.
    // Example 4:

    // Input: heights = [2,2,2,2]
    // Output: [3]
    // Explanation: Buildings cannot see the ocean if there are buildings of the
    // same height to its right.

    public int[] findBuildings(int[] heights) {

        int n = heights.length;
        List<Integer> list = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            // If the current building is taller,
            // it will block the shorter building's ocean view to its left.
            // So we pop all the shorter buildings that have been added before.
            while (!list.isEmpty() && heights[list.get(list.size() - 1)] <= heights[i]) {
                list.remove(list.size() - 1);
            }
            list.add(i);
        }

        // Push elements from list to answer array.
        int[] answer = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            answer[i] = list.get(i);
        }

        return answer;
    }

    // 636. Exclusive Time of Functions
    // On a single-threaded CPU, we execute a program containing n functions. Each
    // function has a unique ID between 0 and n-1.

    // Function calls are stored in a call stack: when a function call starts, its
    // ID is pushed onto the stack, and when a function call ends, its ID is popped
    // off the stack. The function whose ID is at the top of the stack is the
    // current function being executed. Each time a function starts or ends, we
    // write a log with the ID, whether it started or ended, and the timestamp.

    // You are given a list logs, where logs[i] represents the ith log message
    // formatted as a string "{function_id}:{"start" | "end"}:{timestamp}". For
    // example, "0:start:3" means a function call with function ID 0 started at the
    // beginning of timestamp 3, and "1:end:2" means a function call with function
    // ID 1 ended at the end of timestamp 2. Note that a function can be called
    // multiple times, possibly recursively.

    // A function's exclusive time is the sum of execution times for all function
    // calls in the program. For example, if a function is called twice, one call
    // executing for 2 time units and another call executing for 1 time unit, the
    // exclusive time is 2 + 1 = 3.

    // Return the exclusive time of each function in an array, where the value at
    // the ith index represents the exclusive time for the function with ID i.

    // Input: n = 2, logs =
    // ["0:start:0","0:start:2","0:end:5","1:start:6","1:end:6","0:end:7"]
    // Output: [7,1]
    // Explanation:
    // Function 0 starts at the beginning of time 0, executes for 2 units of time,
    // and recursively calls itself.
    // Function 0 (recursive call) starts at the beginning of time 2 and executes
    // for 4 units of time.
    // Function 0 (initial call) resumes execution then immediately calls function
    // 1.
    // Function 1 starts at the beginning of time 6, executes 1 units of time, and
    // ends at the end of time 6.
    // Function 0 resumes execution at the beginning of time 6 and executes for 2
    // units of time.
    // So function 0 spends 2 + 4 + 1 = 7 units of total time executing, and
    // function 1 spends 1 unit of total time executing.

    public int[] exclusiveTime(int n, List<String> logs) {
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();
        int prevTime = 0; // pre means the start of the interval
        for (String str : logs) {
            String[] chunks = str.split(":");
            if (chunks[1].equals("start")) {
                // chunks[2] is the start of next interval, doesn't belong to current interval.
                if (!stack.isEmpty())
                    result[stack.peek()] += Integer.parseInt(chunks[2]) - prevTime;
                stack.push(Integer.parseInt(chunks[0]));
                prevTime = Integer.parseInt(chunks[2]);
            } else {
                // chunks[2] is end of current interval, belong to current interval. That's why
                // we have +1 here
                result[stack.pop()] += Integer.parseInt(chunks[2]) - prevTime + 1;
                // prevTime means the start of next interval, so we need to +1
                prevTime = Integer.parseInt(chunks[2]) + 1;
            }
        }
        return result;
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

    // Example 1:
    // Input: accounts =
    // [["John","johnsmith@mail.com","john_newyork@mail.com"],["John","johnsmith@mail.com","john00@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
    // Output:
    // [["John","john00@mail.com","john_newyork@mail.com","johnsmith@mail.com"],["Mary","mary@mail.com"],["John","johnnybravo@mail.com"]]
    // Explanation:
    // The first and second John's are the same person as they have the common email
    // "johnsmith@mail.com".
    // The third John and Mary are different people as none of their email addresses
    // are used by other accounts.
    // We could return these lists in any order, for example the answer [['Mary',
    // 'mary@mail.com'], ['John', 'johnnybravo@mail.com'],
    // ['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']]
    // would still be accepted.

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

    // 269. Alien Dictionary
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

    // 408. Valid Word Abbreviation
    // A string can be abbreviated by replacing any number of non-adjacent,
    // non-empty substrings with their lengths. The lengths should not have leading
    // zeros.

    // For example, a string such as "substitution" could be abbreviated as (but not
    // limited to):

    // "s10n" ("s ubstitutio n")
    // "sub4u4" ("sub stit u tion")
    // "12" ("substitution")
    // "su3i1u2on" ("su bst i t u ti on")
    // "substitution" (no substrings replaced)

    // Example 1:

    // Input: word = "internationalization", abbr = "i12iz4n"
    // Output: true
    // Explanation: The word "internationalization" can be abbreviated as "i12iz4n"
    // ("i nternational iz atio n").
    // Example 2:

    // Input: word = "apple", abbr = "a2e"
    // Output: false
    // Explanation: The word "apple" cannot be abbreviated as "a2e".

    public boolean validWordAbbreviation(String word, String abbr) {
        if (word == null || abbr == null)
            return false;
        int i = 0, j = 0;

        while (i < word.length() && j < abbr.length()) {
            char c1 = word.charAt(i);
            char c2 = abbr.charAt(j);

            if (c1 == c2) {
                i++;
                j++;
            } else if (Character.isDigit(c2) && c2 != '0') {
                int skip = 0;
                while (j < abbr.length() && Character.isDigit(abbr.charAt(j))) {
                    skip = skip * 10 + (abbr.charAt(j) - '0');
                    j++;
                }
                i += skip;
            } else {
                return false;
            }
        }
        return i == word.length() && j == abbr.length();
    }

    // 21. Merge Two Sorted Lists

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

    // 242. Valid Anagram

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

    // 1. Two Sum
    // Given an array of integers nums and an integer target, return indices of the
    // two numbers such that they add up to target.

    // You may assume that each input would have exactly one solution, and you may
    // not use the same element twice.
    // You can return the answer in any order.

    // Input: nums = [2,7,11,15], target = 9
    // Output: [0,1]
    // Output: Because nums[0] + nums[1] == 9, we return [0, 1].

    // 122. Best Time to Buy and Sell Stock II

    public int maxProfit2(int[] prices) {
        if (prices == null || prices.length == 0)
            return 0;
        int profit = 0;
        for (int i = 0; i < prices.length - 1; i++) {
            if (prices[i + 1] > prices[i]) {
                profit += prices[i + 1] - prices[i];
            }
        }
        return profit;
    }

    public int[] twoSum1(int[] nums, int target) {
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

    // 283. Move Zeroes

    // Given an integer array nums, move all 0's to the end of it while maintaining
    // the relative order of the non-zero elements.

    // Note that you must do this in-place without making a copy of the array.

    // Example 1:

    // Input: nums = [0,1,0,3,12]
    // Output: [1,3,12,0,0]

    public void moveZeros1(int[] nums) {
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if (nums[i] != 0) {
                nums[index++] = nums[i];
            }
        }

        for (; index < nums.length;) {
            nums[index++] = 0;
        }
    }

    // 278. First Bad Version

    // You are a product manager and currently leading a team to develop a new
    // product. Unfortunately, the latest version of your product fails the quality
    // check. Since each version is developed based on the previous version, all the
    // versions after a bad version are also bad.

    // Suppose you have n versions [1, 2, ..., n] and you want to find out the first
    // bad one, which causes all the following ones to be bad.

    // You are given an API bool isBadVersion(version) which returns whether version
    // is bad. Implement a function to find the first bad version. You should
    // minimize the number of calls to the API.

    public int firstBadVersion2(int n) {
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (isBadVersion(mid)) {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        return left;
    }

    // 403 Frog Jump

    // A frog is crossing a river. The river is divided into some number of units,
    // and at each unit, there may or may not exist a stone. The frog can jump on a
    // stone, but it must not jump into the water.

    // Given a list of stones' positions (in units) in sorted ascending order,
    // determine if the frog can cross the river by landing on the last stone.
    // Initially, the frog is on the first stone and assumes the first jump must be
    // 1 unit.

    // If the frog's last jump was k units, its next jump must be either k - 1, k,
    // or k + 1 units. The frog can only jump in the forward direction.

    // Example 1:

    // Input: stones = [0,1,3,5,6,8,12,17]
    // Output: true
    // Explanation: The frog can jump to the last stone by jumping 1 unit to the 2nd
    // stone, then 2 units to the 3rd stone, then 2 units to the 4th stone, then 3
    // units to the 6th stone, 4 units to the 7th stone, and 5 units to the 8th
    // stone.
    // Example 2:

    // Input: stones = [0,1,2,3,4,8,9,11]
    // Output: false
    // Explanation: There is no way to jump to the last stone as the gap between the
    // 5th and 6th stone is too large.

    public boolean canCross(int[] stones) {
        for (int i = 3; i < stones.length; i++) {
            if (stones[i] > stones[i - 1] * 2)
                return false;
        }
        Set<Integer> stonePositions = new HashSet<>();
        for (int s : stones)
            stonePositions.add(s);

        int lastStone = stones[stones.length - 1];
        Stack<Integer> positions = new Stack<>();
        Stack<Integer> jumps = new Stack<>();
        positions.add(0);
        jumps.add(0);
        while (!positions.isEmpty()) {
            int position = positions.pop();
            int jumpDistance = jumps.pop();
            for (int i = jumpDistance - 1; i <= jumpDistance + 1; i++) {
                if (i <= 0)
                    continue;
                int nextPosition = position + i;
                if (nextPosition == lastStone)
                    return true;
                else if (stonePositions.contains(nextPosition)) {
                    positions.add(nextPosition);
                    jumps.add(i);
                }
            }
        }
        return false;
    }

    // 62. Unique Paths

    // A robot is located at the top-left corner of a m x n grid (marked 'Start' in
    // the diagram below).

    // The robot can only move either down or right at any point in time. The robot
    // is trying to reach the bottom-right corner of the grid (marked 'Finish' in
    // the diagram below).

    // How many possible unique paths are there?

    // Example 1:

    // Input: m = 3, n = 7
    // Output: 28

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

    // 286. Walls and Gates

    // You are given an m x n grid rooms initialized with these three possible
    // values.

    // -1 A wall or an obstacle.
    // 0 A gate.
    // INF Infinity means an empty room. We use the value

    // Fill each empty room with the distance to its nearest gate. If it is
    // impossible to reach a gate, it should be filled with INF.

    public void wallsAndGates(int[][] rooms) {
        for (int i = 0; i < rooms.length; i++) {
            for (int j = 0; j < rooms[i].length; j++) {
                if (rooms[i][j] == 0) {
                    dfsWallsAndGates(i, j, 0, rooms);
                }
            }
        }
    }

    private void dfsWallsAndGates(int i, int j, int distance, int[][] rooms) {
        if (i < 0 || i >= rooms.length || j < 0 || j >= rooms[i].length || rooms[i][j] < distance) {
            return;
        }

        rooms[i][j] = distance;
        dfsWallsAndGates(i + 1, j, distance + 1, rooms);
        dfsWallsAndGates(i - 1, j, distance + 1, rooms);
        dfsWallsAndGates(i, j + 1, distance + 1, rooms);
        dfsWallsAndGates(i, j - 1, distance + 1, rooms);
    }

    // 202. Happy Number

    // Write an algorithm to determine if a number n is happy.

    // A happy number is a number defined by the following process:

    // Starting with any positive integer, replace the number by the sum of the
    // squares of its digits.
    // Repeat the process until the number equals 1 (where it will stay), or it
    // loops endlessly in a cycle which does not include 1.
    // Those numbers for which this process ends in 1 are happy.
    // Return true if n is a happy number, and false if not.

    // Example 1:

    // Input: n = 19
    // Output: true
    // Explanation:
    // 12 + 92 = 82
    // 82 + 22 = 68
    // 62 + 82 = 100
    // 12 + 02 + 02 = 1

    public boolean isHappy(int n) {
        Set<Integer> seen = new HashSet<>();
        while (n != 1) {
            int current = n;
            int sum = 0;
            while (current != 0) {
                sum += (current % 10) * (current % 10);
                current /= 10;
            }
            if (seen.contains(sum))
                return false;
            seen.add(sum);
            n = sum;
        }
        return true;
    }

    // 14. Longest Common Prefix

    // Write a function to find the longest common prefix string amongst an array of
    // strings.

    // If there is no common prefix, return an empty string "".

    // Example 1:
    // Input: strs = ["flower","flow","flight"]
    // Output: "fl"

    // Example 2:
    // Input: strs = ["dog","racecar","car"]
    // Output: ""
    // Explanation: There is no common prefix among the input strings.

    public String longestCommonPrefix(String[] strs) {
        String longestCommonPrefix = "";
        if (strs == null || strs.length == 0)
            return longestCommonPrefix;

        int index = 0;
        for (char c : strs[0].toCharArray()) {
            for (int i = 1; i < strs.length; i++) {
                if (index >= strs[i].length() || c != strs[i].charAt(index)) {
                    return longestCommonPrefix;
                }
            }
            longestCommonPrefix += c;
            index++;
        }
        return longestCommonPrefix;
    }

    // 78. Subsets

    // Given an integer array nums of unique elements, return all possible subsets
    // (the power set).

    // The solution set must not contain duplicate subsets. Return the solution in
    // any order.

    // Example 1:

    // Input: nums = [1,2,3]
    // Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> subsets = new ArrayList<>();
        generateSubsets(0, nums, new ArrayList<Integer>(), subsets);
        return subsets;
    }

    private void generateSubsets(int index, int[] nums, List<Integer> current, List<List<Integer>> subsets) {
        subsets.add(new ArrayList<>(current));
        for (int i = index; i < nums.length; i++) {
            current.add(nums[i]);
            generateSubsets(i + 1, nums, current, subsets);
            current.remove(current.size() - 1);
        }
    }

    // 416. Partition Equal Subset Sum

    // Given a non-empty array nums containing only positive integers, find if the
    // array can be partitioned into two subsets such that the sum of elements in
    // both subsets is equal.

    // Example 1:

    // Input: nums = [1,5,11,5]
    // Output: true
    // Explanation: The array can be partitioned as [1, 5, 5] and [11].

    public boolean canPartition(int[] nums) {
        int total = 0;
        for (int i : nums) {
            total += i;
        }

        if (total % 2 != 0)
            return false;

        return dfsCanPartition(nums, 0, 0, total, new HashMap<String, Boolean>());
    }

    public boolean dfsCanPartition(int[] nums, int index, int sum, int total, Map<String, Boolean> state) {
        String current = index + "" + sum;
        if (state.containsKey(current)) // dp
            return state.get(current);
        if (sum * 2 == total)
            return true;

        if (sum > total / 2 || index >= nums.length)
            return false;

        boolean foundPartition = dfsCanPartition(nums, index + 1, sum, total, state)
                || dfsCanPartition(nums, index + 1, sum + nums[index], total, state);
        state.put(current, foundPartition);
        return foundPartition;
    }

    // 257. Binary Tree Paths

    // Given the root of a binary tree, return all root-to-leaf paths in any order.

    // A leaf is a node with no children.

    // Example 1:

    // Input: root = [1,2,3,null,5]
    // Output: ["1->2->5","1->3"]

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> paths = new ArrayList<>();
        if (root == null)
            return paths;
        dfsBinaryTreePaths(root, "", paths);
        return paths;
    }

    private void dfsBinaryTreePaths(TreeNode node, String path, List<String> paths) {
        path += node.val;
        if (node.left == null && node.right == null) {
            paths.add(path);
            return;
        }
        if (node.left != null)
            dfsBinaryTreePaths(node.left, path + "->", paths);

        if (node.right != null)
            dfsBinaryTreePaths(node.right, path + "->", paths);
    }

    // 938. Range Sum of BST

    // Given the root node of a binary search tree and two integers low and high,
    // return the sum of values of all nodes with a value in the inclusive range
    // [low, high].

    // Input: root = [10,5,15,3,7,null,18], low = 7, high = 15
    // Output: 32
    // Explanation: Nodes 7, 10, and 15 are in the range [7, 15]. 7 + 10 + 15 = 32.

    int totalSum;

    public int rangeSumBST1(TreeNode root, int L, int R) {
        totalSum = 0;
        dfs(root, L, R);
        return totalSum;
    }

    public void dfsRangeSumBST(TreeNode node, int L, int R) {
        if (node != null) {
            if (L <= node.val && node.val <= R)
                totalSum += node.val;
            if (L < node.val)
                dfs(node.left, L, R);
            if (node.val < R)
                dfs(node.right, L, R);
        }
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

    // 415. Add Strings

    // Given two non-negative integers, num1 and num2 represented as string, return
    // the sum of num1 and num2 as a string.

    // You must solve the problem without using any built-in library for handling
    // large integers (such as BigInteger). You must also not convert the inputs to
    // integers directly.

    // Example 1:

    // Input: num1 = "11", num2 = "123"
    // Output: "134"

    public String addStrings1(String num1, String num2) {
        int carry = 0;
        StringBuilder res = new StringBuilder();
        int p1 = num1.length() - 1;
        int p2 = num2.length() - 1;
        while (p1 >= 0 || p2 >= 0) {
            int x1 = p1 >= 0 ? num1.charAt(p1) - '0' : 0;
            int x2 = p2 >= 0 ? num2.charAt(p2) - '0' : 0;
            int sum = (x1 + x2 + carry) % 10;
            carry = (x1 + x2 + carry) / 10;
            res.insert(0, sum);
            p1--;
            p2--;
        }
        if (carry != 0) {
            res.insert(0, carry);
        }
        return res.toString();
    }

    // 767. Reorganize String

    // Given a string s, rearrange the characters of s so that any two adjacent
    // characters are not the same.

    // Return any possible rearrangement of s or return "" if not possible.

    // Example 1:

    // Input: s = "aab"
    // Output: "aba"
    // Example 2:

    // Input: s = "aaab"
    // Output: ""

    public String reorganizeString(String s) {
        Map<Character, Integer> counts = new HashMap<>();
        for (char c : s.toCharArray()) {
            counts.put(c, counts.getOrDefault(c, 0) + 1);
        }

        PriorityQueue<Character> maxHeap = new PriorityQueue<>((a, b) -> counts.get(b) - counts.get(a));
        maxHeap.addAll(counts.keySet());

        StringBuilder result = new StringBuilder();
        while (maxHeap.size() > 1) {
            char current = maxHeap.remove();
            char next = maxHeap.remove();
            result.append(current);
            result.append(next);

            counts.put(current, counts.get(current) - 1);
            counts.put(next, counts.get(next) - 1);

            if (counts.get(current) > 0) {
                maxHeap.add(current);
            }

            if (counts.get(next) > 0) {
                maxHeap.add(next);
            }
        }

        if (!maxHeap.isEmpty()) {
            char last = maxHeap.remove();
            if (counts.get(last) > 1)
                return "";
            result.append(last);
        }
        return result.toString();
    }

    // 1108. Defanging an IP Address

    // Given a valid (IPv4) IP address, return a defanged version of that IP
    // address.

    // A defanged IP address replaces every period "." with "[.]".

    // Example 1:
    // Input: address = "1.1.1.1"
    // Output: "1[.]1[.]1[.]1"

    // Example 2:
    // Input: address = "255.100.50.0"
    // Output: "255[.]100[.]50[.]0"

    public String defangIPaddr(String address) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < address.length(); i++) {
            char current = address.charAt(i);
            if (current == '.')
                result.append("[.]");
            else
                result.append(current);
        }
        return result.toString();
    }

    // 298. Binary Tree Longest Consecutive Sequence

    // Given the root of a binary tree, return the length of the longest consecutive
    // sequence path.

    // The path refers to any sequence of nodes from some starting node to any node
    // in the tree along the parent-child connections. The longest consecutive path
    // needs to be from parent to child (cannot be the reverse).

    // Input: root = [1,null,3,2,4,null,null,null,5]
    // Output: 3
    // Explanation: Longest consecutive sequence path is 3-4-5, so return 3.

    public int longestConsecutive(TreeNode root) {
        int[] max = new int[1];
        dfsLongestConsecutive(root, 0, 0, max);
        return max[0];
    }

    private void dfsLongestConsecutive(TreeNode node, int count, int target, int[] max) {
        if (node == null)
            return;
        else if (node.val == target)
            count++;
        else
            count = 1;
        max[0] = Math.max(max[0], count);
        dfsLongestConsecutive(node.left, count, node.val + 1, max);
        dfsLongestConsecutive(node.right, count, node.val + 1, max);
    }

    // 443. String Compression

    // Given an array of characters chars, compress it using the following
    // algorithm:

    // Begin with an empty string s. For each group of consecutive repeating
    // characters in chars:

    // If the group's length is 1, append the character to s.
    // Otherwise, append the character followed by the group's length.
    // The compressed string s should not be returned separately, but instead, be
    // stored in the input character array chars. Note that group lengths that are
    // 10 or longer will be split into multiple characters in chars.

    // After you are done modifying the input array, return the new length of the
    // array.

    // You must write an algorithm that uses only constant extra space.

    // Example 1:

    // Input: chars = ["a","a","b","b","c","c","c"]
    // Output: Return 6, and the first 6 characters of the input array should be:
    // ["a","2","b","2","c","3"]
    // Explanation: The groups are "aa", "bb", and "ccc". This compresses to
    // "a2b2c3".

    public int compress(char[] chars) {
        int index = 0;
        int i = 0;
        while (i < chars.length) {
            int j = i;
            while (j < chars.length && chars[j] == chars[i])
                j++;

            chars[index++] = chars[i];

            if (j - i > 1) {
                String count = j - i + "";
                for (char c : count.toCharArray()) {
                    chars[index++] = c;
                }
            }
            i = j;
        }
        return index;
    }
}
