package com.clinicalprediction.backend.dto;

import jakarta.validation.constraints.*;

public record PredictionRequest(
    @NotNull @Min(0) @Max(120) Integer age,
    @NotBlank String gender,
    @NotNull @Min(0) @Max(3) Integer phq1,
    @NotNull @Min(0) @Max(3) Integer phq2,
    @NotNull @Min(0) @Max(3) Integer phq3,
    @NotNull @Min(0) @Max(3) Integer phq4,
    @NotNull @Min(0) @Max(3) Integer phq5,
    @NotNull @Min(0) @Max(3) Integer phq6,
    @NotNull @Min(0) @Max(3) Integer phq7,
    @NotNull @Min(0) @Max(3) Integer phq8,
    @NotNull @Min(0) @Max(3) Integer phq9
) {}

