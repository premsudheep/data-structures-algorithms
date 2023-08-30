package com.prem;

import java.util.Map;
import java.util.HashMap;
import java.util.NoSuchElementException;
import java.util.Stack;

public class LinkedList {
  private class Node {
    private int value;
    private Node next;

    public Node(int value) {
      this.value = value;
    }
  }

  private Node first;
  private Node last;
  private int size;

  public void addLast(int item) {
    var node = new Node(item);

    if (isEmpty())
      first = last = node;
    else {
      last.next = node;
      last = node;
    }

    size++;
  }

  public void addFirst(int item) {
    var node = new Node(item);

    if (isEmpty())
      first = last = node;
    else {
      node.next = first;
      first = node;
    }

    size++;
  }

  private boolean isEmpty() {
    return first == null;
  }

  public int indexOf(int item) {
    int index = 0;
    var current = first;
    while (current != null) {
      if (current.value == item)
        return index;
      current = current.next;
      index++;
    }
    return -1;
  }

  public boolean contains(int item) {
    return indexOf(item) != -1;
  }

  public void removeFirst() {
    if (isEmpty())
      throw new NoSuchElementException();

    if (first == last)
      first = last = null;
    else {
      var second = first.next;
      first.next = null;
      first = second;
    }

    size--;
  }

  public void removeLast() {
    if (isEmpty())
      throw new NoSuchElementException();

    if (first == last)
      first = last = null;
    else {
      var previous = getPrevious(last);
      last = previous;
      last.next = null;
    }

    size--;
  }

  private Node getPrevious(Node node) {
    var current = first;
    while (current != null) {
      if (current.next == node)
        return current;
      current = current.next;
    }
    return null;
  }

  public int size() {
    return size;
  }

  public int[] toArray() {
    int[] array = new int[size];
    var current = first;
    var index = 0;
    while (current != null) {
      array[index++] = current.value;
      current = current.next;
    }

    return array;
  }

  public void reverse() {
    if (isEmpty())
      return;

    var previous = first;
    var current = first;
    while (current != null) {
      var next = current.next;
      current.next = previous;
      previous = current;
      current = next;
    }

    last = first;
    last.next = null;
    first = previous;
  }

  public int getKthFromTheEnd(int k) {
    if (isEmpty())
      throw new IllegalStateException();

    var a = first;
    var b = first;
    for (int i = 0; i < k - 1; i++) {
      b = b.next;
      if (b == null)
        throw new IllegalArgumentException();
    }
    while (b != last) {
      a = a.next;
      b = b.next;
    }
    return a.value;
  }

  public void printMiddle() {
    if (isEmpty())
      throw new IllegalStateException();

    var a = first;
    var b = first;
    while (b != last && b.next != last) {
      b = b.next.next;
      a = a.next;
    }

    if (b == last)
      System.out.println(a.value);
    else
      System.out.println(a.value + ", " + a.next.value);
  }

  public boolean hasLoop() {
    var slow = first;
    var fast = first;

    while (fast != null && fast.next != null) {
      slow = slow.next;
      fast = fast.next.next;

      if (slow == fast)
        return true;
    }

    return false;
  }

  public static LinkedList createWithLoop() {
    var list = new LinkedList();
    list.addLast(10);
    list.addLast(20);
    list.addLast(30);

    // Get a reference to 30
    var node = list.last;

    list.addLast(40);
    list.addLast(50);

    // Create the loop
    list.last.next = node;

    return list;
  }

  public boolean isPalindrome(Node head) {
    Node fast = head;
    Node slow = head;
    while (fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    fast = head;
    slow = reverse(slow);
    while (slow != null) {
      if (slow.value != fast.value)
        return false;
      slow = slow.next;
      fast = fast.next;
    }
    return true;
  }

  public Node reverse(Node head) {
    if (head == null)
      return null;
    Node previous = null;
    Node current = head;
    while (current != null) {
      var next = current.next;
      current.next = previous;
      previous = current;
      current = next;
    }
    return previous;
  }

  public Node removeElements(Node head, int val) {
    if (head == null)
      return null;
    Node newNode = new Node(-1);
    newNode.next = head;
    head = newNode;
    while (newNode.next != null) {
      if (newNode.next.value == val) {
        newNode.next = newNode.next.next;
      } else {
        newNode = newNode.next;
      }
    }
    return head.next;
  }

  public Node intersection(Node a, Node b) {
    // return the node that is intersecting
    // a->b
    //      \
    //        d->e->null //return d node
    //      /
    // b->c 

    Node pointerA = a;
    Node pointerB = b;
    while (pointerA != pointerB) {
      pointerA = pointerA.next;
      pointerB = pointerB.next;
      if (pointerA == pointerB)
        return pointerA;
      if (pointerA == null)
        pointerA = b;
      if (pointerB == null)
        pointerB = a;
    }
    return pointerA; // or pointerB;
  }

  public Node detectCycle(Node head) {
    if (head == null || head.next == null)
      return null;

    Node slow = head;
    Node fast = head;
    while (true) {
      slow = slow.next;
      fast = fast.next.next;
      if (fast == null || fast.next == null)
        return null;
      if (slow == fast)
        break;
    }
    slow = head;
    while (slow != fast) {
      if (slow == fast)
        return slow;
      slow = slow.next;
      fast = fast.next;
    }
    return slow;// or fast
  }

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

  // 19. Remove Nth Node From End of List

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

  // 24. Swap Nodes in Pairs

  // Given a linked list, swap every two adjacent nodes and return its head. You
  // must solve the problem without modifying the values in the list's nodes
  // (i.e., only nodes themselves may be changed.)

  public ListNode swapPairs(ListNode head) {

    // Dummy node acts as the prevNode for the head node
    // of the list and hence stores pointer to the head node.
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

  // OR

  public ListNode swapPairsUsingStack(ListNode head) {
    Stack<ListNode> stack = new Stack<>();
    stack.push(null);
    ListNode current = head;
    while (current != null) {
      if (stack.size() == 2) {
        ListNode next = current.next;
        ListNode prior = stack.pop();
        ListNode priorPrior = stack.pop();
        if (priorPrior != null)
          priorPrior.next = current;
        else
          head = current;
        current.next = prior;
        prior.next = next;
        current = prior;
      }
      stack.push(current);
      current = current.next;
    }
    return head;
  }

  // 138. Copy List with Random Pointer

  // Literally Deep copy the node and return

  class NodeRandom {
    int val;
    NodeRandom next;
    NodeRandom random;

    public NodeRandom(int val) {
      this.val = val;
      this.next = null;
      this.random = null;
    }
  }

  Map<NodeRandom, NodeRandom> visited = new HashMap<>();

  public NodeRandom copyRandomList(NodeRandom head) {
    if (head == null)
      return null;

    if (visited.containsKey(head))
      return visited.get(head);

    NodeRandom node = new NodeRandom(head.val);
    visited.put(head, node);

    node.next = copyRandomList(head.next);
    node.random = copyRandomList(head.random);

    return node;
  }

  // 92. Reverse Linked List II
  // Given the head of a singly linked list and two integers left and right where
  // left <= right, reverse the nodes of the list from position left to position
  // right, and return the reversed list.

  // Input: head = [1->2->3->4->5], left = 2, right = 4
  // Output: [1->4->3->2->5]

  public ListNode reverseBetween(ListNode head, int left, int right) {
    // Empty list
    if (head == null)
      return null;

    // Move the two pointers until they reach the proper starting point in the list.
    ListNode current = head;
    ListNode prev = null;
    while (left > 1) {
      prev = current;
      current = current.next;
      left--;
      right--;
    }

    // The two pointers that will fix the final connections.
    ListNode conn = prev;
    ListNode tail = current;

    // Iteratively reverse the nodes until n becomes 0.
    ListNode next = null;
    while (right > 0) {
      next = current.next;
      current.next = prev;
      prev = current;
      current = next;
      right--;
    }

    // Adjust the final connections as explained in the algorithm
    if (conn != null) {
      conn.next = prev;
    } else {
      head = prev;
    }

    tail.next = current;
    return head;
  }

  // 445. Add Two Numbers II
  // Example 1:
  // Input: l1 = [7,2,4,3], l2 = [5,6,4]
  // Output: [7,8,0,7]
  // Example 2:
  // Input: l1 = [2,4,3], l2 = [5,6,4]
  // Output: [8,0,7]

  // Reverse the Input and add them back and build in reverse order

  public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
    l1 = reverseLinkedList(l1);
    l2 = reverseLinkedList(l2);

    ListNode head = null;
    int carry = 0;
    while (l1 != null || l2 != null) {
      int v1 = l1 != null ? l1.val : 0;
      int v2 = l2 != null ? l2.val : 0;

      int sum = (carry + v1 + v2);
      int value = sum % 10;
      carry = sum / 10;

      ListNode current = new ListNode(value);
      current.next = head;
      head = current;

      l1 = l1 != null ? l1.next : null;
      l2 = l2 != null ? l2.next : null;
    }

    if (carry != 0) {
      ListNode current = new ListNode(carry);
      current.next = head;
      head = current;
    }

    return head;

  }

  private ListNode reverseLinkedList(ListNode head) {
    ListNode prev = null;
    ListNode current = head;
    while (current != null) {
      ListNode next = current.next;
      current.next = prev;
      prev = current;
      current = next;
    }

    return prev;
  }

  // 328. Odd Even Linked List
  // Given the head of a singly linked list, group all the nodes with odd indices
  // together followed by the nodes with even indices, and return the reordered
  // list.
  // Input: head = [2,1,3,5,6,4,7]
  // Output: [2,3,6,7,1,5,4]

  public ListNode oddEvenList(ListNode head) {
    if (head == null)
      return null;
    ListNode oddHead = head;
    ListNode evenHead = oddHead.next;
    ListNode odd = oddHead;
    ListNode even = evenHead;

    while (even != null && even.next != null) {
      odd.next = even.next;
      odd = odd.next;
      even.next = odd.next;
      even = even.next;
    }
    odd.next = evenHead;
    return oddHead;
  }

}
