package com.prem;

import java.util.*;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Stack;

public class LeetCode {

    private class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int val) { this.val = val; }
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
        // s = anagram, t = nagaram, return true. rearrangement of characters. lower case only
        if(s.length() != t.length())
            return false;
        int[] counts = new int[26];
        for(int i = 0; i < s.length(); i++) {
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
        for(int i = 0; i < s.length(); i++) {
            char current = s.charAt(i);
            if(!map.containsKey(current))
                map.put(current, i);
            else
                map.put(current, -1);
        }
        int min = Integer.MAX_VALUE;
        for (char c : map.keySet()) {
            if(map.get(c) > -1 && map.get(c) < min)
                min = map.get(c);
        }
        return min == Integer.MAX_VALUE ? -1 : min;
    }

    public int buySellStocks(int[] prices) {
        // Best time to buy and sell stocks [7, 1, 5, 3, 6, 4] -> 6 - 1 = 5, buy on day 2 and sell on day 5
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
        //[1,2,3,1]->true
        Set<Integer> set = new HashSet<>();
        for(int num : nums) {
            if(set.contains(num))
                return true;
            else
                set.add(num);
        }
        return false;
    }

    public List<String> fizzBuzz(int n) {
        List<String> result = new ArrayList<>();
        for(int i = 1; i <= n; i++) {
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
        for(char c : s.toCharArray()) {
            if(c == '(' || c == '{' || c == '[')
                stack.push(c);
            else if(c == ')' && !stack.isEmpty() && stack.peek() == '(')
                stack.pop();
            else if(c == '}' && !stack.isEmpty() && stack.peek() == '{')
                stack.pop();
            else if(c == ']' && !stack.isEmpty() && stack.peek() == '[')
                stack.pop();
            else
                return false;
        }
        return stack.isEmpty();
    }

    public int singleNumber(int[] nums) {
        // every list element appears twice except one -> find that number. [2,2,1]
        Set<Integer> set = new HashSet<>();
        for(int i : nums) {
            if(set.contains(i))
                set.remove(i);
            else
                set.add(i);
        }

        return set.iterator().next();
    }

    public boolean powerOfTwo(int n) {
        //input 1 -> 2 to the power of 0 is 1, so return true. 16 -> 2 to the power of 4 is 16, so return true.
        long i = 1;
        while (i < n) {
            i *= 2;
        }
        return i == n;
    }

    public int numberOfIslands(char[][] grid) {
        if(grid == null || grid.length == 0)
            return 0;
        int result = 0;
        for(int i = 0; i < grid.length; i++) {
            for (int j = 0 ; j < grid[i].length; j++) {
                if(grid[i][j] == '1'){
                    result += islandsDFS(grid, i, j);
                }
            }
        }
        return result;
    }

    private int islandsDFS(char[][] grid, int i, int j) {
        if(i < 0 || i >= grid.length || j < 0 || j >= grid[i].length || grid[i][j] == 0)
            return 0;
        islandsDFS(grid, i+1, j);
        islandsDFS(grid, i-1, j);
        islandsDFS(grid, i, j+1);
        islandsDFS(grid, i, j-1);
        return 1;
    }

    public int findPeakElement(int[] nums) {
        // element that is greater than its neighbours
        //[1,2,3,1] -> 3's index 2
        // faster solution is binary search, array has to be sorted
        int left = 0;
        int right = nums.length - 1;
        while(left < right) {
            int mid = left + (right - left) / 2;
            if(nums[mid] < nums[mid + 1])
                left = mid + 1;
            else
                right = mid;
        }
        return left;
    }

    public int[] twoSum(int[] nums, int target) {
        // return indices of two nums such that they add to a specific target [2, 7, 11, 15] target = 9 -> return [0, 1]
        int[] result = new int[2];
        Map<Integer, Integer> map = new HashMap<>();
        for(int  i = 0; i< nums.length; i++) {
            int diff = target - nums[i];
            if(map.containsKey(diff)) {
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
        for(char ch : s.toCharArray()) {
            if(ch != '#') {
                sStack.push(ch);
            } else if(!sStack.empty()) {
                sStack.pop();
            }
        }
        for(char ch : t.toCharArray()) {
            if(ch != '#') {
                tStack.push(ch);
            } else if(!sStack.empty()) {
                tStack.pop();
            }
        }
        while(!sStack.isEmpty()) {
            char current = sStack.pop();
            if(!tStack.isEmpty() || tStack.pop() != current)
                return false;
        }
        return tStack.isEmpty();
    }

    public int[] moveZeros(int[] nums) {
        // move zeros to the end of the array [0,1,0,3,12] -> [1,3,12,0,0]
        // have an index and move the value to left once at the end then fill the rest with zeros
        int index = 0;
        for (int i = 0; i < nums.length; i++) {
            if(nums[i] != 0)
                nums[index++] = nums[i];
        }
        for( int i = index; i < nums.length; i++) {
            nums[i] = 0;
        }
        return nums;
    }

    public int reverseInteger(int x) {
        // 32 bit signed int 123 -> 321, -123 -> -321
        boolean negative = false;
        if(x < 0) {
            negative = true;
            x *= -1;
        }
        long reversed = 0;
        while(x > 0) {
            reversed  = (reversed * 10) + (x % 10);
        }
        if(reversed > Integer.MAX_VALUE)
            return -1;
        return negative ? (int)(reversed * -1) : (int) reversed;
    }

    public int[] plusOne(int[] digits) {
        //[4,3,2,1] -> [4,3,2,2]
        //[4,3,2,9] -> [4,3,3,0]
        for(int i = digits.length - 1; i >= 0; i--) {
            if(digits[i] < 9) {
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

    public void firstBadVersion(int n) {
        // Given n = 5 and version = 4 is the first bad version
        //call isBadVersion(3) -> false
        //call isBadVersion(4) -> true
        //call isBadVersion(5) -> true
        // better solution is a binary search // 0001111111
        int left = 1;
        int right = n;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if(!isBadVersion(mid))
                left = mid + 1;
            else
                right = mid;
        }
    }

    private boolean isBadVersion(int x) { return false; }

    public int paintHouse(int[][] costs) {
        // [[17,2,17],[16,16,5],[14,3,19]] -> output 10.
        // paint house 0 into blue
        // paint house 1 into green
        // paint house 2 into blue
        // find the min cost to paint house
        if(costs == null || costs.length == 0)
            return 0;
        for (int i = 1; i< costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        return Math.min(Math.min(costs[costs.length - 1][0], costs[costs.length - 1][1]), costs[costs.length -1][2]);
    }

    public boolean robotReturnToOrigin(String moves) {
        // UD -> true
        // LL -> false
        // judgeCircle
        int UD = 0;
        int LR = 0;
        for(int i = 0; i < moves.length(); i++) {
            char currentMove = moves.charAt(i);
            if(currentMove == 'U')
                UD++;
            else if (currentMove == 'D')
                UD--;
            else if(currentMove == 'L')
                LR++;
            else if (currentMove == 'R')
                LR--;
        }
        return UD == 0 && LR == 0;
    }

    public boolean containsDuplicate(int[] nums, int k) {
        //[1,2,3,1], k = 3 -> true
        //[1,2,3,1,2,3], k = 2 -> false
        // k is the difference of duplicates
        Map<Integer, Integer> map = new HashMap<>();
        for(int i = 0; i < nums.length; i++) {
            int current = nums[i];
            if(map.containsKey(current) && i - map.get(current) <= k)
                return true;
            else
                map.put(current, i);
        }
        return false;
    }

    public char findTheDifference(String s, String t) {
        // s=abcd, t=abcde, return e
        Map<Character, Integer> map = new HashMap<>();
        for(char c : s.toCharArray())
            map.put(c, map.getOrDefault(c, 0)+1);
        for(char c : t.toCharArray()) {
            if((map.containsKey(c) && map.get(c) == 0) || !map.containsKey(c))
                return c;
            else
                map.put(c, map.get(c) - 1);
        }
        return ' ';
    }

    public TreeNode convertSortedArrayToBST(int[] nums) {
        // [-10, -3, 0, 5, 9] -> [0, -3, 9, -10, null, 5]
        if(nums == null || nums.length == 0)
            return null;
        return constructBSTRecursive(nums, 0, nums.length - 1);
    }

    private TreeNode constructBSTRecursive(int[] nums, int left, int right) {
        if(left > right)
            return null;
        int mid = left + (right - left) / 2;
        TreeNode current = new TreeNode(nums[mid]);
        current.left = constructBSTRecursive(nums, left, mid - 1);
        current.right = constructBSTRecursive(nums, mid + 1, right);
        return current;

    }

    public int findTheCelebrity(int n) {
        // celeb -> all the n-1 ppl in the part know him/her but he/she don't know any one of them
        int person = 0;
        for(int i = 1; i < n; i++) {
            if(knows(person, i)) {
                person = i;
            }
        }
        for(int i = 0; i < n; i++) {
            if(i != person && knows(person, i) || !knows(i, person))
                return -1;
        }
        return person;
    }

    private boolean knows(int a, int b) {
        //this helper will return true if either a knows b but b dont know a and vice versa.
        return true;
    }

    public int[] intersectionOfTwoArrays(int[] nums1, int[] nums2) {
        // [1,2,2,1] [2,2] -> [2]
        Set<Integer> set = new HashSet<>();
        for(int num : nums1) {
            set.add(num);
        }
        Set<Integer> intersection = new HashSet<>();
        for(int num : nums2) {
            if(set.contains(num))
                intersection.add(num);
        }

        int[] result = new int[intersection.size()];
        int index = 0;
        for(int i : intersection)
            result[index++] = i;
        return result;
    }

    public int findTheMissingNumber(int[] nums) {
        // [3, 0 ,1] -> return 2
        // array {0, 1, 2,...n} n distinct elements
        // Gauss's rule/law [n(n-1)/2]
        int sum = 0;
        for(int i : nums)
            sum += i;
        int n = nums.length + 1; // includes 0
        return ((n*(n-1))/2) - sum;
    }

    public boolean meetingRooms(Interval[] intervals) {
        //[[0,30][5,10][15,20]] -> false
        //[[7,10][2,4]] -> true
        // determine if a person could attend all meetings.
        int[] start = new int[intervals.length];
        int[] end = new int[intervals.length];
        for(int i = 0; i < intervals.length; i++) {
            start[i] = intervals[i].start;
            end[i] = intervals[i].end;
        }
        Arrays.sort(start);
        Arrays.sort(end);
        for(int i = 0; i < start.length; i++) {
            if(start[i + 1] < end[i])
                return false;
        }
        return true;
    }

    class Interval {
        int start;
        int end;
        Interval() {start = 0; end = 0;}
        Interval(int s, int e) {this.start = s; this.end = e;}
    }

    public int majorityElement(int[] nums) {
        // find the element that appears more than half the time
        // [3,2,3] -> 3
        if(nums.length == 1)
            return nums[0];
        Map<Integer, Integer> map = new HashMap<>();
        for(int i : nums) {
            if(map.containsKey(i) && map.get(i) + 1 > nums.length/2)
                return i;
            else
                map.put(i, map.getOrDefault(i, 0)+1);
        }
        return -1;
    }

    public int hammingDistance(int x, int y) {
        // dist between two integers is the number of positions at which corresponding bits are different.
        // x=1, y=4 -> 2
        // 1 (0 0 0 1)
        // 4 (0 1 0 0)
        // above rows the intersection bits are different at 2,4 columns (2 different places the bits are not same. so, return 2)
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
        //[7,1,5,3,6,4] -> buy on day 2(1) and sell on day 3(5) and buy on day 4(3) and sell on day 5(6)
        if(prices == null || prices.length == 0)
            return 0;
        int profit = 0;
        for(int i = 0; i < prices.length; i++) {
            if(prices[i + 1] > prices[i])
                profit += prices[i + 1] - prices[i];
        }
        return profit;
    }

    public int singleNumber2(int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        for(int i : nums)
            map.put(i, map.getOrDefault(i , 0) + 1);
        for(int key : map.keySet())
            if(map.get(key) == 1)
                return key;
        return -1;
    }

    public boolean pathSum(TreeNode root, int sum) {
        // given BST, sum the root to leaf such that equals to given sum
        if(root == null)
            return false;
        else if(root.left == null && root.right == null && sum - root.val == 0)
            return true;
        else
            return pathSum(root.left, sum - root.val) || pathSum(root.right, sum - root.val);
    }

    public int removeElement(int[] nums, int val) {
        int index = 0;
        for(int i : nums) {
            if(i != val)
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
        // how many ways we can climb the n stairs (at a time can step either 1 or 2 steps)
        // input = 2 -> (1+1, 2 steps) -> output = 2
        // input = 3 -> (1+1+1, 1+2, 2+1 steps) -> output = 3
        int[] dp = new int[n + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= n; i++){
            dp[i] = dp[i - 1] + dp[i -2];
        }
        return dp[n];
    }

    public List<String> phoneNumberLetterCombinations(String digits) {
        // input -> 23-> [ad,ae, af, bd, be, bf, cd, ce, cf]
        List<String> result = new ArrayList<>();
        if(digits == null || digits.length() == 0)
            return result;
        String[] mappings = {
                "0",
                "1",
                "abc",
                "def",
                "ghi",
                "jkl",
                "mno",
                "pqrs",
                "tuv",
                "wxyz"
        };
        combo(result, digits, "", 0, mappings);
        return result;
    }

    private void combo(List<String> result, String digits, String current, int index, String[] mappings) {
        if(index == digits.length()) {
            result.add(current);
            return;
        }
        String letters = mappings[digits.charAt(index)-'0'];
        for(int i=0; i<letters.length(); i++) {
            combo(result, digits, current + letters.charAt(i), index + 1, mappings);
        }
    }

    public TreeNode lowestCommonAncestorInBST(TreeNode root, TreeNode p, TreeNode q) {
        if(p.val < root.val && q.val < root.val)
            return lowestCommonAncestorInBST(root.left, p , q);
        else if(p.val > root.val && q.val > root.val)
            return lowestCommonAncestorInBST(root.right, p , q);
        else
            return root;
    }

    public int jewelsStones(String j, String s) {
        Set<Character> set = new HashSet<>();
        for(char c : j.toCharArray())
            set.add(c);
        int numJewel = 0;
        for (char c : s.toCharArray())
            if(set.contains(c))
                numJewel++;
        return numJewel;
    }

    public boolean binaryNumberWithAlternatingBits(int n) {
        // 5 -> 101 true, 7 -> 111 false, 2 -> 010 true, 10 -> 1010 true, 11 -> 1011 -> false
        int last = n % 2;
        n >>= 1; // diving by 2
        while(n > 0) {
            int current = n % 2;
            if(current == last)
                return false;
            last = current;
            n >>= 1;
        }
        return true;
    }

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

    // 40 completed

    public int containerWithMostWater(int[] height) {
        /*int max = Integer.MIN_VALUE;
        for(int i=0;i<height.length;i++){
            for(int j=i+1; j<height.length;j++){
                int min = Math.min(height[i], height[j]); // restriction to overflow the water
                max = Math.max(max, min * (j - i));
            }
        }
        return max;*/

        int max = Integer.MIN_VALUE;
        int i =0;
        int j = height.length - 1;
        while (i < j) {
            int min = Math.min(height[i], height[j]); // restriction to overflow the water
            max = Math.max(max, min * (j - i));
            if(height[i]<height[j])
                i++;
            else
                j--;
        }
        return max;
    }

    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null)
            return 0;
        else if(root.left != null && root.left.left == null && root.left.right == null)
            return root.left.val + sumOfLeftLeaves(root.right);
        else
            return sumOfLeftLeaves(root.left) + sumOfLeftLeaves(root.right);

    }

    public List<Integer> bstPostOrderTraversal(TreeNode root) {
        List<Integer> values = new ArrayList<>();
        if(root == null)
            return values;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()) {
            TreeNode current = stack.pop();
            values.add(0, current.val); // always push the existing values and add to the 1st position
            if(current.left != null)
                stack.push(current.left);
            if(current.right != null)
                stack.push(current.right);
        }
        return values;
    }

    public int battleShipsInBoard(char[][] board) {
        int numBattleShips = 0;
        for(int i =0; i< board.length; i++) {
            for (int j = 0; j < board[i].length; j++){
                //Optimized
                if(board[i][j] == '.')
                    continue;
                if(i > 0 && board[i - 1][j] == 'X')
                    continue;
                if(j > 0 && board[i][j - 1] == 'X')
                    continue;
                numBattleShips++;
            }
        }
        return numBattleShips;
    }

    private void sinkShips(char[][] board, int i, int j) {
        if(i<0||i>=board.length||j<0||j>board[i].length||board[i][j]!='X')
            return;

        board[i][j] = '.';
        sinkShips(board, i+1, j);
        sinkShips(board, i-1, j);
        sinkShips(board, i, j+1);
        sinkShips(board, i, j-1);

    }

    public int[][] flippingImage(int[][] A) {
        for(int i =0; i<A.length;i++) {
            int j =0;
            int k = A[i].length - 1;
            while(j<k){
                int temp = A[i][j];
                A[i][j++] = A[i][k];
                A[i][k--] = temp;
            }

            for(j = 0; j<A[i].length; j++)
                A[i][j] = A[i][j] == 1 ? 0 : 1;
        }
        return A;
    }

    public int kthLargestElementInArray(int[] nums, int k) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int i : nums) {
            minHeap.add(i);
            if(minHeap.size() > k)
                minHeap.remove();
        }
        return minHeap.remove();
    }

    public int totalFruit(int[] tree){
        // longest substring with k(2) characters
        if(tree == null || tree.length == 0)
            return 0;
        int max = 1;
        Map<Integer, Integer> map = new HashMap<>();
        int i = 0;
        int j = 0;
        while (j < tree.length) {
            if(map.size() <= 2) {
                map.put(tree[j], j++);
            }
            if(map.size() > 2) {
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

    public int numDecodings(String s) {
        //'A' -> 1
        //'B' -> 2
        //'Z' -> 26
        // '12' -> A,B,L -> 3 occurrence
        int[] dp = new int[s.length() + 1];
        dp[0] = 1;
        dp[1] = s.charAt(0) == '0' ? 0 : 1;
        for (int i=2; i<=s.length(); i++) {
            int oneDigit = Integer.valueOf(s.substring(i-1, i));
            int twoDigit = Integer.valueOf(s.substring(i-2, i));
            if(oneDigit >= 1) {
                dp[i] += dp[i-1];
            }
            if(twoDigit >= 10 && twoDigit <=26) {
                dp[i] += dp[i-2];
            }
        }
        return dp[s.length()];
    }

    public List<List<Integer>> binaryTreeLevelOrderTraversal(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        if(root == null)
            return result;
        Queue<TreeNode> q = new LinkedList<>();
        q.add(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> currentLevel = new ArrayList<>();
            for(int i=0;i<size;i++){
                TreeNode current = q.remove();
                currentLevel.add(current.val);
                if(current.left != null)
                    q.add(current.left);
                if(current.right != null)
                    q.add(current.right);
            }
            result.add(currentLevel);
         }
        return result;
    }

    public int meetingRooms2(Interval[] intervals) {
        if(intervals == null || intervals.length == 0)
            return 0;
        Arrays.sort(intervals, (a,b) -> a.start - b.start);
        PriorityQueue<Interval> minHeap = new PriorityQueue<>((a,b) -> a.end - b.end);
        minHeap.add(intervals[0]);
        for(int i=1;i<intervals.length;i++) {
            Interval current  = intervals[i];
            Interval earliest = minHeap.remove();
            if(current.start >= earliest.end)
                earliest.end = current.end;
            else
                minHeap.add(current);
            minHeap.add(earliest);
        }
        return minHeap.size();
    }

    public int[][] kClosestPoint(int[][] points, int k) {
        // input [[1,3][-2,2]] k = 1 -> output [[-2,2]]
        PriorityQueue<int[]> maxHeap = new PriorityQueue<>((a,b) ->
                (b[0]*b[0]+b[1]*b[1])-(a[0]*a[0]+a[1]*a[1]));
        //d = sqrt((x2-x1)+(y2-y1))
        for(int[] point: points) {
            maxHeap.add(point);
            if(maxHeap.size() > k)
                maxHeap.remove();
        }
        int[][] result = new int[k][2];
        while (k-->0) {
            result[k] = maxHeap.remove();
        }
        return result;
    }

    public String sortCharactersByFrequency(String s) {
        Map<Character, Integer> map = new HashMap<>();
        for(char c:s.toCharArray())
            map.put(c, map.getOrDefault(c,0)+1);
        PriorityQueue maxHeap = new PriorityQueue((a,b)->map.get(b)-map.get(a));
        maxHeap.addAll(map.keySet());
        StringBuilder sb = new StringBuilder();
        while (!maxHeap.isEmpty()) {
            char current = (char) maxHeap.remove();
            for(int i=0;i<map.get(current);i++) {
                sb.append(current);
            }
        }
        return sb.toString();
    }

}
