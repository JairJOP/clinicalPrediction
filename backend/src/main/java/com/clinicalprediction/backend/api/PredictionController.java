package com.clinicalprediction.backend.api;


import java.util.List;

import org.springframework.web.server.ResponseStatusException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.web.client.RestTemplateBuilder;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestTemplate;

import com.clinicalprediction.backend.dto.FeedbackRequest;
import com.clinicalprediction.backend.dto.PredictionRequest;
import com.clinicalprediction.backend.dto.PredictionResponse;
import com.clinicalprediction.backend.model.PredictionRecord;
import com.clinicalprediction.backend.repo.PredictionRepository;

import jakarta.validation.Valid;



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
                mlUrl + "/predict?model=" + req.model(),
                req,
                PredictionResponse.class
        );

        if (mlOut == null) {
            throw new ResponseStatusException(HttpStatus.BAD_GATEWAY, "ML server returned null response");
        }

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
        rec.setModel(req.model());

    
        rec.setPrediction(mlOut.prediction());
        rec.setProbability(mlOut.probability());
        rec.setThreshold(mlOut.threshold());
        rec.setModel(mlOut.model());
        

        rec = repo.save(rec);
        mlOut = new PredictionResponse(
            rec.getId(),
            mlOut.prediction(),
            mlOut.probability(),
            mlOut.threshold(),
            mlOut.model()
        );

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

    @GetMapping("/predictions")
    public List<PredictionRecord> getRecentPredictions() {
    return repo.findAll(PageRequest.of(0, 20, Sort.by(Sort.Direction.DESC, "createdAt"))).getContent();
    }

    @GetMapping("/predictions/{id}")
    public ResponseEntity<PredictionRecord> getPredictionById(@PathVariable Long id) {
    return repo.findById(id)
            .map(ResponseEntity::ok)
            .orElse(ResponseEntity.notFound().build());
    }

    @PostMapping("/feedback")
    public ResponseEntity<String> submitFeedback(@RequestBody FeedbackRequest feedback) {
        Long id = feedback.getPredictionId();
        String message = feedback.getMessage();

        return repo.findById(id)
            .map(record -> {
                record.setFeedback(message);
                repo.save(record);
                return ResponseEntity.ok("Feedback saved.");
            })
            .orElse(ResponseEntity.notFound().build());
    }

}