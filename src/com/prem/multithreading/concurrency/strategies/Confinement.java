package com.prem.multithreading.concurrency.strategies;

import java.util.*;

public class Confinement {
    public static void main(String[] args) {

        List<Thread> threads = new ArrayList<>();
        List<DownloadFileTask> tasks = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            DownloadFileTask task = new DownloadFileTask();
            tasks.add(task);

            Thread thread = new Thread(task);
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

        Optional<Integer> totalBytes = tasks.stream().map(t -> t.getStatus().getTotalBytes()).reduce(Integer::sum);
        System.out.println(totalBytes.get());
    }
}

class DownloadFileTask implements Runnable {

    private DownloadStatus status;

    public DownloadFileTask() {
        this.status = new DownloadStatus();
    }

    public DownloadStatus getStatus() {
        return status;
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

class DownloadStatus {
    private int totalBytes;

    public int getTotalBytes() {
        return totalBytes;
    }

    public void incrementTotalBytes() {
        totalBytes++;
    }

}
