package com.prem;

import java.util.Arrays;
import java.util.*;
import java.util.PriorityQueue;

class Main {

    public static void main(String[] args) {
        String[] input = { "Selfie Stick,98,29", "iPhone Case,90,15", "Fire TV Stick,48,49", "Wyze Cam,48,25",
                "Water Filter,56,49", "Blue Light Blocking Glasses,90,16", "Ice Maker,47,119", "Video Doorbell,47,199",
                "AA Batteries,64,12", "Disinfecting Wipes,37,12", "Baseball Cards,73,16", "Winter Gloves,32,112",
                "Microphone,44,22", "Pet Kennel,5,24", "Jenga Classic Game,100,7", "Ink Cartridges,88,45",
                "Instant Pot,98,59", "Hoze Nozzle,74,26", "Gift Card,45,25", "Keyboard,82,19" };
        List<String> inputList = Arrays.asList(input);
        traverse(inputList);

    }

    private static void traverse(List<String> list) {
        Queue<String> nodeEntries = new PriorityQueue<>((e1, e2) -> {
            String[] s1 = e1.split(",");
            String[] s2 = e2.split(",");
            // 1 -> pop
            // 2 -> price
            for (int i = 1; i < s1.length; i++) {
                if (i == 1 && Integer.parseInt(s1[i]) != Integer.parseInt(s2[i]))
                    return Integer.parseInt(s2[i]) - Integer.parseInt(s1[i]);
                else
                    return Integer.parseInt(s1[i]) - Integer.parseInt(s2[i]);
            }
            return 0;

        });

        for (String s : list) {
            nodeEntries.add(s);
        }

        while (!nodeEntries.isEmpty()) {
            System.out.println(nodeEntries.remove().split(",")[0]);
        }
    }
}
