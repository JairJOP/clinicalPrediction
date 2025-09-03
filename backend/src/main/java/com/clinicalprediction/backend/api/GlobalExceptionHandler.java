package com.clinicalprediction.backend.api;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.client.RestClientException;

import java.time.Instant;
import java.util.Map;

@ControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(RestClientException.class)
    public ResponseEntity<?> handleRestClientException(RestClientException ex) {
        return ResponseEntity.status(503).body(Map.of(
                "timestamp", Instant.now(),
                "status", 503,
                "error", "ML service unavailable",
                "message", ex.getMessage()
        ));
    }

    // Optional: fallback for all other exceptions
    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> handleOtherExceptions(Exception ex) {
        return ResponseEntity.status(500).body(Map.of(
                "timestamp", Instant.now(),
                "status", 500,
                "error", "Unexpected server error",
                "message", ex.getMessage()
        ));
    }
}
