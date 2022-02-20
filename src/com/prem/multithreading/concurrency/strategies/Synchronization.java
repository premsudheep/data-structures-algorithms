package com.prem.multithreading.concurrency.strategies;

import java.util.*;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Synchronization {
    public static void main(String[] args) {

        // // Fixing with the synchronization
        // DownloadStatus1 status = new DownloadStatus1();
        // List<Thread> threads = new ArrayList<>();
        // for (int i = 0; i < 10; i++) {
        // Thread thread = new Thread(new DownloadFileTask1(status));
        // thread.start();
        // threads.add(thread);
        // }
        // for (Thread thread : threads) {
        // try {
        // thread.join();
        // } catch (InterruptedException e) {
        // e.printStackTrace();
        // }
        // }
        // System.out.println(status.getTotalBytes());

        // Fixing the Visibility problem
        DownloadStatus1 status1 = new DownloadStatus1();
        Thread thread1 = new Thread(new DownloadFileTask1(status1));
        Thread thread2 = new Thread(() -> {
            // Thread Signalling
            synchronized (status1) {
                while (!status1.isDone()) {
                    try {
                        status1.wait();// This goes to sleep until another thread wakes it up.
                    } catch (InterruptedException e) {
                        e.printStackTrace();
                    }
                }
            }
            System.out.println(status1.getTotalBytes());
        });

        thread1.start();
        thread2.start();
    }
}

class DownloadFileTask1 implements Runnable {

    private DownloadStatus1 status;

    public DownloadFileTask1(DownloadStatus1 status) {
        this.status = status;
    }

    @Override
    public void run() {
        System.out.println("Downloading a file: " + Thread.currentThread().getName());
        for (int i = 0; i < 1_000_000; i++) {
            if (Thread.currentThread().interrupted())
                return;
            status.incrementTotalBytes();
        }

        // visibility fix
        status.done();
        // singnal the waiting thread to resume when done.
        synchronized (status) {
            status.notifyAll();
        }
        System.out.println("Downloading completed: " + Thread.currentThread().getName());
    }

}

class DownloadStatus1 {
    private volatile boolean isDone; // Fixes the visibility problem (meaning don't read from cache always read form
                                     // main memory)
    private int totalBytes;
    Object totalBytesObject = new Object(); // Monitor Object, has to be unique.

    public int getTotalBytes() {
        return totalBytes;
    }

    public void incrementTotalBytes() {
        synchronized (totalBytesObject) { // fixes the race condition
            totalBytes++;
        }
    }

    public boolean isDone() {
        return isDone;
    }

    public void done() {
        isDone = true;
    }

}
