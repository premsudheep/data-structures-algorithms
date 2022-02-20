package com.prem.multithreading.executiveframework;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;

public class CompletableFutureDemo {
    public static void main(String[] args) {
        // ForkJoinPool.commonPool() --> if we dont suppy apool, then CompletableFuture
        // use this common executor
        Runnable task1 = () -> System.out.println("a");
        CompletableFuture.runAsync(task1);

        Supplier<Integer> task2 = () -> 1;
        CompletableFuture<Integer> future = CompletableFuture.supplyAsync(task2);
        try {
            int result = future.get();
            System.out.println(result);

            future.thenRunAsync(() -> {
                System.out.println(Thread.currentThread().getName()); // this will run on seperate thread
                System.out.println("Done");
            });

            // has to pass a consumer object
            future.thenAccept(action -> {
                System.out.println(Thread.currentThread().getName()); // this will be executed on the main thread.
                System.out.println(action);
            });

            // This will submit the task to executor asynchronously.
            future.thenAcceptAsync(action -> {
                System.out.println(Thread.currentThread().getName()); // this will be executed on the main thread.
                System.out.println(action);
            });

            Thread.sleep(3000);

        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // 2 Exception Handling.
        CompletableFuture<Integer> futureException = CompletableFuture.supplyAsync(() -> {
            System.out.println("Getting the current weather"); // Weather service
            throw new IllegalArgumentException();
        });
        try {
            int value = futureException.exceptionally(ex -> 1).get(); // this will return a new CompleteableFuture
            System.out.println(value);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // 3. Transformation of Completable Future
        CompletableFuture<Integer> futureTransformable = CompletableFuture.supplyAsync(() -> 20);
        try {
            Double fahrenheit = futureTransformable.thenApply(celsius -> (celsius * 1.8) + 32).get(); // this will
                                                                                                      // return a
            // CompletableFuture
            System.out.println(fahrenheit);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        // OR
        CompletableFuture.supplyAsync(() -> 20).thenApply(c -> (c * 1.8) + 32).thenAccept(f -> System.out.println(f));

        // 4. Composing Completable Future
        // Imagine we have a DB that has email and want to get that and send to a music
        // service.
        // id -> email
        CompletableFuture<String> futureEmailId = CompletableFuture.supplyAsync(() -> "email");
        // email -> playlist
        futureEmailId.thenCompose(email -> CompletableFuture.supplyAsync(() -> "playlist"))
                .thenAccept(playlist -> System.out.println(playlist));

        // OR
        CompletableFuture.supplyAsync(() -> "email")
                .thenCompose(email -> CompletableFuture.supplyAsync(() -> "playlist"))
                .thenAccept(playlist -> System.out.println(playlist));

        // 5. Combining Completable Future.
        CompletableFuture<Integer> amount = CompletableFuture.supplyAsync(() -> "20USD").thenApply(str -> {
            String price = str.replace("USD", "");
            return Integer.parseInt(price);
        });
        CompletableFuture<Double> currencyConversionRate = CompletableFuture.supplyAsync(() -> 0.9);

        amount.thenCombine(currencyConversionRate, (price, exchangeRate) -> price * exchangeRate)
                .thenAccept(result -> System.out.println(result));

        // 6. Waiting for Many Tasks to Complete.
        CompletableFuture<Integer> first = CompletableFuture.supplyAsync(() -> 1);
        CompletableFuture<Integer> second = CompletableFuture.supplyAsync(() -> 2);
        CompletableFuture<Integer> third = CompletableFuture.supplyAsync(() -> 3);

        CompletableFuture<Void> all = CompletableFuture.allOf(first, second, third);
        all.thenRun(() -> {
            try {
                int firstResult = first.get();
                System.out.println(firstResult);
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
            System.out.println("All tasks completed successfully");
        });

        // 7. Waiting for First Task to Complete.
        CompletableFuture<Integer> one = CompletableFuture.supplyAsync(() -> {
            LongTask.simulate();
            return 20;
        });
        CompletableFuture<Integer> two = CompletableFuture.supplyAsync(() -> 30);

        CompletableFuture<Object> any = CompletableFuture.anyOf(one, two);
        any.thenAccept(temp -> System.out.println(temp));

        // 8. Handling Timeouts.
        CompletableFuture<Integer> longtask = CompletableFuture.supplyAsync(() -> {
            LongTask.simulate();
            return 1;
        });
        try {
            int value = longtask.completeOnTimeout(10, 1, TimeUnit.SECONDS).get();
            System.out.println(value);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

    }

}
