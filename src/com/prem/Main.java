package com.prem;

import java.util.Arrays;
import java.util.*;
import java.util.HashMap;
import java.util.Set;
import java.util.TreeSet;
import java.util.Stack;

public class Main {
    public static void main(String[] names) {
        int a = 2, b = -3;
        while (b != 0) {
            int answer = a ^ b;
            int carry = (a & b) << 1;
            a = answer;
            b = carry;
        }
        System.out.println(Character.toString((char) 254));
    }

}
