package com.prem.multithreading.executiveframework;

import java.util.concurrent.*;

public class ExecutorDemo {
    public static void main(String[] args) {
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        try {
            Future<Integer> future = executor.submit(() -> { // submit method will return immidiately
                LongTask.simulate(); // E.g. call twitter API to get some tweets
                return 1;
            });
            System.out.println("Do more work.!");

            int result = future.get(); // when get is called, then Main thread has to wait until it is LongTask task is
                                       // completed.
            System.out.println(result);
        } catch (InterruptedException | ExecutionException e) {

        } finally {
            executor.shutdown();
        }
    }
}

class LongTask {
    public static void simulate() {
        try {
            Thread.sleep(3000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    public static void simulate(int delay) {
        try {
            Thread.sleep(delay);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
