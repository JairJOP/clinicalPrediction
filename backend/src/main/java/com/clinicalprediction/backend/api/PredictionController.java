package com.clinicalprediction.backend.api;

import com.clinicalprediction.backend.dto.PredictionRequest;
import com.clinicalprediction.backend.dto.PredictionResponse;
import com.clinicalprediction.backend.model.PredictionRecord;
import com.clinicalprediction.backend.repo.PredictionRepository;

import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.web.client.RestTemplate;

@RestController
@RequestMapping("/api")
public class PredictionController {

    private final RestTemplate rest;
    private final PredictionRepository repo;

    @Value("${ml.url}")
    private String mlUrl; // e.g. http://127.0.0.1:8000

    public PredictionController(RestTemplateBuilder builder, PredictionRepository repo) {
        this.rest = builder.build();
        this.repo = repo;
    }

    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("backend ok");
    }

    @PostMapping(value = "/predict", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public PredictionResponse predict(@Valid @RequestBody PredictionRequest req) {
        // 1) call the Flask ML service
        PredictionResponse mlOut = rest.postForObject(
                mlUrl + "/predict",
                req,
                PredictionResponse.class
        );

        // 2) persist the request+response
        PredictionRecord rec = new PredictionRecord();
        rec.setAge(req.age());
        rec.setGender(req.gender());
        rec.setPhq1(req.phq1());
        rec.setPhq2(req.phq2());
        rec.setPhq3(req.phq3());
        rec.setPhq4(req.phq4());
        rec.setPhq5(req.phq5());
        rec.setPhq6(req.phq6());
        rec.setPhq7(req.phq7());
        rec.setPhq8(req.phq8());
        rec.setPhq9(req.phq9());

        if (mlOut != null) {
            rec.setPrediction(mlOut.prediction());
            rec.setProbability(mlOut.probability());
            rec.setThreshold(mlOut.threshold());
            rec.setModel(mlOut.model());
        }
        repo.save(rec);

        // 3) return the ML response to the client
        return mlOut;
    }

    @PostMapping(value = "/explain", consumes = MediaType.APPLICATION_JSON_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    public Object explain(@Valid @RequestBody PredictionRequest req) {
        // call Flask explain endpoint
        return rest.postForObject(
                mlUrl + "/explain",
                req,
                Object.class
        );
    }
}