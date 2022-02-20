package com.prem.multithreading.executiveframework;

import java.util.concurrent.CompletableFuture;

public class AsynchronousAPIDemo {
    public static void main(String[] args) {
        MailService service = new MailService();
        service.sendAsync();
        System.out.println("Hello world");

        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class MailService {
    public void send() {
        LongTask.simulate();
        System.out.println("Mail was sent.");
    }

    public CompletableFuture<Void> sendAsync() {
        return CompletableFuture.runAsync(() -> send());
    }
}
