package com.clinicalprediction.backend.dto;

public class FeedbackRequest {
    private Long predictionId;
    private String message;

    public Long getPredictionId() {
        return predictionId;
    }

    public void setPredictionId(Long predictionId) {
        this.predictionId = predictionId;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }
}