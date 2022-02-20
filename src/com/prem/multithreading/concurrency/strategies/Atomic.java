package com.prem.multithreading.concurrency.strategies;

import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.LongAdder;

public class Atomic {
    // One way to achieve concurreny with out any race condition is by implementing
    // Atomic classes
    public static void main(String[] args) {
        DownloadStaus2 status = new DownloadStaus2();

        List<Thread> threads = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            Thread thread = new Thread(new DownloadFileTask2(status));
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
        System.out.println(status.getTotalFiles());
    }
}

class DownloadFileTask2 implements Runnable {

    private DownloadStaus2 status;

    public DownloadFileTask2(DownloadStaus2 status) {
        this.status = status;
    }

    @Override
    public void run() {
        System.out.println("Downloading a file: " + Thread.currentThread().getName());
        for (int i = 0; i < 10_000; i++) {
            if (Thread.currentThread().interrupted())
                return;
            status.incrementTotalBytes();
            status.incrementTotalFiles();
        }
        System.out.println("Downloading completed: " + Thread.currentThread().getName());
    }

}

class DownloadStaus2 {
    private AtomicInteger totalBytes = new AtomicInteger(); // Atomic object
    private LongAdder totalFiles = new LongAdder(); // Adders are faster that Atomic Objects

    public int getTotalBytes() {
        return totalBytes.get();
    }

    public void incrementTotalBytes() {
        totalBytes.incrementAndGet();
    }

    public int getTotalFiles() {
        return totalFiles.intValue();
    }

    public void incrementTotalFiles() {
        totalFiles.increment();
    }

}
