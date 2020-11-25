package com.prem;

import java.util.NoSuchElementException;

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
      if (current.value == item) return index;
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
      if (current.next == node) return current;
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
    if (isEmpty()) return;

    var previous = first;
    var current = first.next;
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
    while(fast != null && fast.next != null) {
      fast = fast.next.next;
      slow = slow.next;
    }
    fast = head;
    slow = reverse(slow);
    while (slow != null) {
      if(slow.value != fast.value) return false;
      slow = slow.next;
      fast = fast.next;
    }
    return true;
  }

  public Node reverse(Node head) {
    if (head == null) return null;
    Node previous = null;
    Node current = head.next;
    while (current != null) {
      var next = current.next;
      current.next = previous;
      previous = current;
      current = next;
    }
    return previous;
  }

  public Node removeElements(Node head, int val) {
    if(head == null) return null;
    Node newNode = new Node(-1);
    newNode.next = head;
    head = newNode;
    while(newNode.next != null) {
      if(newNode.next.value == val) {
        newNode.next = newNode.next.next;
      } else {
        newNode = newNode.next;
      }
    }
    return head.next;
  }

  public Node intersection(Node a, Node b) {
    //return the node that is intersecting
    // a->b
    //      \
    //        d->e->null  //return d node
    // b->c /

    Node pointerA = a;
    Node pointerB = b;
    while(pointerA != pointerB) {
      pointerA = pointerA.next;
      pointerB = pointerB.next;
      if(pointerA == pointerB)
        return pointerA;
      if (pointerA == null)
        pointerA = b;
      if (pointerB == null)
        pointerB = a;
    }
    return pointerA; //or pointerB;
  }

  public Node detectCycle(Node head) {
    if(head == null || head.next == null)
      return null;

    Node slow = head;
    Node fast = head;
    while(true) {
      slow = slow.next;
      fast = fast.next.next;
      if(fast == null || fast.next == null)
        return null;
      if(slow == fast)
        break;
    }
    slow = head;
    while(slow != fast) {
      if(slow == fast)
        return slow;
      slow = slow.next;
      fast = fast.next;
    }
    return slow;//or fast
  }
}
