package com.prem.multithreading.executiveframework;

import java.time.Duration;
import java.time.LocalTime;
import java.util.*;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;
import java.util.stream.Stream;

// Fetch price quotes from diffenent sites asynchronously and display them.
public class Exercise {
    public static void main(String[] args) {
        var start = LocalTime.now();
        FlightService service = new FlightService();
        List<CompletableFuture<Void>> futures = service.getQuotes()
                .map(future -> future.thenAccept(System.out::println)).collect(Collectors.toList());
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).thenRun(() -> {
            var end = LocalTime.now();
            var duration = Duration.between(start, end);
            System.out.println("Retrieved all quotes in " + duration.toMillis() + "msec.");
        });

        try {
            Thread.sleep(10_000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}

class Quote {
    private final int price;
    private final String site;

    public Quote(int price, String site) {
        this.price = price;
        this.site = site;
    }

    public int getPrice() {
        return price;
    }

    public String getSite() {
        return site;
    }

    @Override
    public String toString() {
        return "Quote{" + "site='" + site + '\'' + ", price=" + price + "}";
    }
}

class FlightService {

    public Stream<CompletableFuture<Quote>> getQuotes() {
        var sites = List.of("site1", "site2", "site3");
        return sites.stream().map(this::getQuote);
    }

    public CompletableFuture<Quote> getQuote(String site) {
        return CompletableFuture.supplyAsync(() -> {
            System.out.println("Getting a Quote from : " + site);
            Random random = new Random();
            LongTask.simulate(1_000 + random.nextInt(2_000));
            int price = 100 + random.nextInt(10); // range -> 100 - 110
            return new Quote(price, site);
        });
    }
}