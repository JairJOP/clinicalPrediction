package com.clinicalprediction.backend.repo;

import com.clinicalprediction.backend.model.PredictionRecord;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PredictionRepository extends JpaRepository<PredictionRecord, Long> {}