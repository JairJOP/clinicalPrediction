package com.clinicalprediction.backend.dto;

public record PredictionResponse(
    int prediction,
    double probability,
    double threshold,
    String model
) {}
