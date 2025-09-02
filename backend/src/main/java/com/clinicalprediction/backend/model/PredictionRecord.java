package com.clinicalprediction.backend.model;

import jakarta.persistence.*;
import java.time.Instant;

@Entity
@Table(name = "prediction_records")
public class PredictionRecord {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    // inputs
    private Integer age;
    private String gender;

    private Integer phq1;
    private Integer phq2;
    private Integer phq3;
    private Integer phq4;
    private Integer phq5;
    private Integer phq6;
    private Integer phq7;
    private Integer phq8;
    private Integer phq9;

    // outputs
    private Integer prediction;
    private Double probability;
    private Double threshold;
    private String model;

    private Instant createdAt;

    public PredictionRecord() {}

    @PrePersist
    @SuppressWarnings("unused")
    void prePersist() {
        if (createdAt == null) createdAt = Instant.now();
    }

    // --- getters & setters ---
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }

    public Integer getAge() { return age; }
    public void setAge(Integer age) { this.age = age; }

    public String getGender() { return gender; }
    public void setGender(String gender) { this.gender = gender; }

    public Integer getPhq1() { return phq1; }
    public void setPhq1(Integer phq1) { this.phq1 = phq1; }

    public Integer getPhq2() { return phq2; }
    public void setPhq2(Integer phq2) { this.phq2 = phq2; }

    public Integer getPhq3() { return phq3; }
    public void setPhq3(Integer phq3) { this.phq3 = phq3; }

    public Integer getPhq4() { return phq4; }
    public void setPhq4(Integer phq4) { this.phq4 = phq4; }

    public Integer getPhq5() { return phq5; }
    public void setPhq5(Integer phq5) { this.phq5 = phq5; }

    public Integer getPhq6() { return phq6; }
    public void setPhq6(Integer phq6) { this.phq6 = phq6; }

    public Integer getPhq7() { return phq7; }
    public void setPhq7(Integer phq7) { this.phq7 = phq7; }

    public Integer getPhq8() { return phq8; }
    public void setPhq8(Integer phq8) { this.phq8 = phq8; }

    public Integer getPhq9() { return phq9; }
    public void setPhq9(Integer phq9) { this.phq9 = phq9; }

    public Integer getPrediction() { return prediction; }
    public void setPrediction(Integer prediction) { this.prediction = prediction; }

    public Double getProbability() { return probability; }
    public void setProbability(Double probability) { this.probability = probability; }

    public Double getThreshold() { return threshold; }
    public void setThreshold(Double threshold) { this.threshold = threshold; }

    public String getModel() { return model; }
    public void setModel(String model) { this.model = model; }

    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
}