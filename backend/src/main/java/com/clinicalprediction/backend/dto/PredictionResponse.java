package com.clinicalprediction.backend.dto;

public record PredictionResponse(
    Long id,
    int prediction,
    double probability,
    double threshold,
    String model
    
) {}
