package com.prem;

import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.TreeSet;
import java.util.LinkedList;

public class Tree {
  private class Node {
    private int value;
    private Node leftChild;
    private Node rightChild;

    public Node(int value) {
      this.value = value;
    }

    @Override
    public String toString() {
      return "Node=" + value;
    }
  }

  private Node root;

  public void insert(int value) {
    var node = new Node(value);

    if (root == null) {
      root = node;
      return;
    }

    var current = root;
    while (true) {
      if (value < current.value) {
        if (current.leftChild == null) {
          current.leftChild = node;
          break;
        }
        current = current.leftChild;
      } else {
        if (current.rightChild == null) {
          current.rightChild = node;
          break;
        }
        current = current.rightChild;
      }
    }
  }

  public void insertBST(int data) {
    // Create a new node
    Node newNode = new Node(data);
    // Check whether tree is empty
    if (root == null) {
      root = newNode;
      return;
    } else {
      // current node point to root of the tree
      Node current = root, parent = null;
      while (true) {
        // parent keep track of the parent node of current node.
        parent = current;
        // If data is less than current's data, node will be inserted to the left of
        // tree
        if (data < current.value) {
          current = current.leftChild;
          if (current == null) {
            parent.leftChild = newNode;
            return;
          }
        }
        // If data is greater than current's data, node will be inserted to the right of
        // tree
        else {
          current = current.rightChild;
          if (current == null) {
            parent.rightChild = newNode;
            return;
          }
        }
      }
    }
  }

  public boolean find(int value) {
    var current = root;
    while (current != null) {
      if (value < current.value)
        current = current.leftChild;
      else if (value > current.value)
        current = current.rightChild;
      else
        return true;
    }
    return false;
  }

  public void traversePreOrder() {
    traversePreOrder(root);
  }

  private void traversePreOrder(Node root) {
    if (root == null)
      return;

    System.out.println(root.value);
    traversePreOrder(root.leftChild);
    traversePreOrder(root.rightChild);
  }

  public void traverseInOrder() {
    traverseInOrder(root);
  }

  private void traverseInOrder(Node root) {
    if (root == null)
      return;

    traverseInOrder(root.leftChild);
    System.out.println(root.value);
    traverseInOrder(root.rightChild);
  }

  public void traversePostOrder() {
    traversePostOrder(root);
  }

  private void traversePostOrder(Node root) {
    if (root == null)
      return;

    traversePostOrder(root.leftChild);
    traversePostOrder(root.rightChild);
    System.out.println(root.value);
  }

  public int height() {
    return height(root);
  }

  private int height(Node root) {
    if (root == null)
      return -1;

    if (isLeaf(root))
      return 0;

    return 1 + Math.max(height(root.leftChild), height(root.rightChild));
  }

  private boolean isLeaf(Node node) {
    return node.leftChild == null && node.rightChild == null;
  }

  // O(log n)
  public int min() {
    if (root == null)
      throw new IllegalStateException();

    var current = root;
    var last = current;
    while (current != null) {
      last = current;
      current = current.leftChild;
    }
    return last.value;
  }

  // O(n)
  private int min(Node root) {
    if (isLeaf(root))
      return root.value;

    var left = min(root.leftChild);
    var right = min(root.rightChild);

    return Math.min(Math.min(left, right), root.value);
  }

  public boolean equals(Tree other) {
    if (other == null)
      return false;

    return equals(root, other.root);
  }

  private boolean equals(Node first, Node second) {
    if (first == null && second == null)
      return true;

    if (first != null && second != null)
      return first.value == second.value && equals(first.leftChild, second.leftChild)
          && equals(first.rightChild, second.rightChild);

    return false;
  }

  public boolean isBinarySearchTree() {
    return isBinarySearchTree(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
  }

  private boolean isBinarySearchTree(Node root, int min, int max) {
    if (root == null)
      return true;

    if (root.value < min || root.value > max)
      return false;

    return isBinarySearchTree(root.leftChild, min, root.value - 1)
        && isBinarySearchTree(root.rightChild, root.value + 1, max);
  }

  public ArrayList<Integer> getNodesAtDistance(int distance) {
    var list = new ArrayList<Integer>();
    getNodesAtDistance(root, distance, list);
    return list;
  }

  private void getNodesAtDistance(Node root, int distance, ArrayList<Integer> list) {
    if (root == null)
      return;

    if (distance == 0) {
      list.add(root.value);
      return;
    }

    getNodesAtDistance(root.leftChild, distance - 1, list);
    getNodesAtDistance(root.rightChild, distance - 1, list);
  }

  public void traverseLevelOrder() {
    for (var i = 0; i <= height(); i++) {
      for (var value : getNodesAtDistance(i))
        System.out.println(value);
    }
  }

  public List<List<Integer>> leveOrderTraversal() {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null)
      return result;

    Queue<Node> queue = new LinkedList<>();
    queue.add(root);

    while (!queue.isEmpty()) {
      int size = queue.size();
      List<Integer> currentLevel = new ArrayList<>();
      for (int i = 0; i < size; i++) {
        Node currentNode = queue.remove();
        currentLevel.add(currentNode.value);
        if (currentNode.leftChild != null)
          queue.add(currentNode.leftChild);
        if (currentNode.rightChild != null)
          queue.add(currentNode.rightChild);
      }
      result.add(currentLevel);
    }
    return result;
  }

  public int size() {
    return size(root);
  }

  private int size(Node root) {
    if (root == null)
      return 0;

    if (isLeaf(root))
      return 1;

    return 1 + size(root.leftChild) + size(root.rightChild);
  }

  public int countLeaves() {
    return countLeaves(root);
  }

  private int countLeaves(Node root) {
    if (root == null)
      return 0;

    if (isLeaf(root))
      return 1;

    return countLeaves(root.leftChild) + countLeaves(root.rightChild);
  }

  public int max() {
    if (root == null)
      throw new IllegalStateException();

    return max(root);
  }

  private int max(Node root) {
    if (root.rightChild == null)
      return root.value;

    return max(root.rightChild);
  }

  public boolean contains(int value) {
    return contains(root, value);
  }

  private boolean contains(Node root, int value) {
    if (root == null)
      return false;

    if (root.value == value)
      return true;

    return contains(root.leftChild, value) || contains(root.rightChild, value);
  }

  public boolean areSibling(int first, int second) {
    return areSibling(root, first, second);
  }

  private boolean areSibling(Node root, int first, int second) {
    if (root == null)
      return false;

    var areSibling = false;
    if (root.leftChild != null && root.rightChild != null) {
      areSibling = (root.leftChild.value == first && root.rightChild.value == second)
          || (root.rightChild.value == first && root.leftChild.value == second);
    }

    return areSibling || areSibling(root.leftChild, first, second) || areSibling(root.rightChild, first, second);
  }

  public List<Integer> getAncestors(int value) {
    var list = new ArrayList<Integer>();
    getAncestors(root, value, list);
    return list;
  }

  private boolean getAncestors(Node root, int value, List<Integer> list) {
    // We should traverse the tree until we find the target value. If
    // find the target value, we return true without adding the current node
    // to the list; otherwise, if we ask for ancestors of 5, 5 will be also
    // added to the list.
    if (root == null)
      return false;

    if (root.value == value)
      return true;

    // If we find the target value in the left or right sub-trees, that means
    // the current node (root) is one of the ancestors. So we add it to the list.
    if (getAncestors(root.leftChild, value, list) || getAncestors(root.rightChild, value, list)) {
      list.add(root.value);
      return true;
    }

    return false;
  }

  public boolean isBalanced() {
    return isBalanced(root);
  }

  private boolean isBalanced(Node root) {
    if (root == null)
      return true;

    var balanceFactor = height(root.leftChild) - height(root.rightChild);

    return Math.abs(balanceFactor) <= 1 && isBalanced(root.leftChild) && isBalanced(root.rightChild);
  }

  public boolean isPerfect() {
    return size() == (Math.pow(2, height() + 1) - 1);
  }

  public int getMinimumDifferenceBwAnyNodeBST(Node root) {
    // The most common idea is to first inOrder traverse the tree and compare the
    // delta between each of the adjacent values. It's guaranteed to have the
    // correct answer because it is a BST thus inOrder traversal values are sorted.
    // Solution 1 - In-Order traverse, time complexity O(N), space complexity O(1).
    return getMinDiffBST(root, Integer.MAX_VALUE, null);

    // return getMinDiffNonBST(root, Integer.MAX_VALUE, new TreeSet<>());
  }

  private int getMinDiffBST(Node currentNode, int min, Node prevNode) {
    // The most common idea is to first inOrder traverse the tree and compare the
    // delta between each of the adjacent values. It's guaranteed to have the
    // correct answer because it is a BST thus inOrder traversal values are sorted.
    // Solution 1 - In-Order traverse, time complexity O(N), space complexity O(1).
    if (root == null)
      return min;
    getMinDiffBST(currentNode.leftChild, min, prevNode);
    if (prevNode != null)
      min = Math.min(min, currentNode.value - prevNode.value);
    prevNode = currentNode;
    getMinDiffBST(currentNode.rightChild, min, prevNode);
    return min;
  }

  private int getMinDiffNonBST(Node currentNode, int min, TreeSet<Integer> set) {
    // What if it is not a BST? (Follow up of the problem) The idea is to put values
    // in a TreeSet and then every time we can use O(lgN) time to lookup for the
    // nearest values.
    // Solution 2 - Pre-Order traverse, time complexity O(NlgN), space complexity
    // O(N).
    if (currentNode == null)
      return min;
    if (!set.isEmpty()) {
      if (set.floor(currentNode.value) != null)
        min = Math.min(min, currentNode.value - set.floor(currentNode.value));
      if (set.ceiling(currentNode.value) != null)
        min = Math.min(min, currentNode.value - set.ceiling(currentNode.value));
      set.add(currentNode.value);
    }
    getMinDiffNonBST(currentNode.leftChild, min, set);
    getMinDiffNonBST(currentNode.rightChild, min, set);
    return min;
  }
}