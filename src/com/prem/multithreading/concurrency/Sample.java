package com.prem.multithreading.concurrency;

import java.util.ArrayList;
import java.util.*;

public class Sample {
    // Prune to end up in a race condition
    public static void main(String[] args) {
        DownloadStaus status = new DownloadStaus();

        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            Thread thread = new Thread(new DownloadFileTask(status));
            thread.start();
            threads.add(thread);
        }

        for (Thread thread : threads) {
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        System.out.println(status.getTotalBytes());
    }
}

class DownloadFileTask implements Runnable {

    private DownloadStaus status;

    public DownloadFileTask(DownloadStaus status) {
        this.status = status;
    }

    @Override
    public void run() {
        System.out.println("Downloading a file: " + Thread.currentThread().getName());
        for (int i = 0; i < 10_000; i++) {
            if (Thread.currentThread().interrupted())
                return;
            status.incrementTotalBytes();
        }
        System.out.println("Downloading completed: " + Thread.currentThread().getName());
    }

}

class DownloadStaus {
    private int totalBytes;

    public int getTotalBytes() {
        return totalBytes;
    }

    public void incrementTotalBytes() {
        totalBytes++;
    }

}
