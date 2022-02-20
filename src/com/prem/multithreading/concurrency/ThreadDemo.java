package com.prem.multithreading.concurrency;

public class ThreadDemo {
    public static void show() {
        System.out.println(Thread.currentThread().getName());

        for (int i = 0; i < 10; i++) {
            Thread thread = new Thread(new DownloadFileTaskDemo());
            thread.start();
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("File is ready to scan");
        }

    }
}

class DownloadFileTaskDemo implements Runnable {

    public DownloadFileTaskDemo() {
    }

    @Override
    public void run() {
        System.out.println("Downloading a file: " + Thread.currentThread().getName());
        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("Downloading completed: " + Thread.currentThread().getName());
    }

}
