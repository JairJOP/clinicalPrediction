package com.clinicalprediction.backend.api;

import com.clinicalprediction.backend.dto.PredictionRequest;
import com.clinicalprediction.backend.dto.PredictionResponse;
import com.clinicalprediction.backend.model.PredictionRecord;
import com.clinicalprediction.backend.repo.PredictionRepository;

import org.junit.jupiter.api.Test;
import org.mockito.Mockito;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.test.web.servlet.MockMvc;
import org.springframework.web.client.RestTemplate;

import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import org.springframework.data.domain.Sort;

import java.util.List;

import java.time.Instant;

import com.fasterxml.jackson.databind.ObjectMapper;

@WebMvcTest(PredictionController.class)
public class PredictionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @SuppressWarnings("removal")
    @MockBean
    private PredictionRepository repo;

    @SuppressWarnings("removal")
    @MockBean
    private RestTemplate rest;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void testPredict() throws Exception {
        PredictionRequest request = new PredictionRequest(25, "male", 1,1,1,1,1,1,1,1,1, "logreg");
        PredictionResponse response = new PredictionResponse(123L, 1, 0.999, 0.25, "logreg");

        Mockito.when(rest.postForObject(Mockito.anyString(), Mockito.any(), Mockito.eq(PredictionResponse.class)))
                .thenReturn(response);

        mockMvc.perform(post("/api/predict")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.prediction").value(1))
                .andExpect(jsonPath("$.probability").value(0.999))
                .andExpect(jsonPath("$.threshold").value(0.25))
                .andExpect(jsonPath("$.model").value("logreg"));
    }

    @Test
    void testExplain() throws Exception {
        // Mock request (same as predict)
        PredictionRequest request = new PredictionRequest(25, "male", 1, 1, 1, 1, 1, 1, 1, 1, 1, "logreg");

        // Mock explanation response (can be anything we assume JSON object)
        Object explainResponse = new Object() {
            public final String explanation = "mock explanation";
        };

        Mockito.when(rest.postForObject(Mockito.anyString(), Mockito.eq(request), Mockito.eq(Object.class)))
              .thenReturn(explainResponse);

        // Perform POST to /api/explain
        mockMvc.perform(post("/api/explain")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(request)))
                .andExpect(status().isOk());
    }

    @Test
    void testGetAllPredictions() throws Exception {
        // Mock database records
        PredictionRecord record1 = new PredictionRecord();
        record1.setId(1L);
        record1.setAge(25);
        record1.setGender("male");
        record1.setPhq1(1);
        record1.setPhq2(1);
        record1.setPhq3(1);
        record1.setPhq4(1);
        record1.setPhq5(1);
        record1.setPhq6(1);
        record1.setPhq7(1);
        record1.setPhq8(1);
        record1.setPhq9(1);
        record1.setPrediction(1);
        record1.setProbability(0.99);
        record1.setThreshold(0.25);
        record1.setModel("logreg");
        record1.setCreatedAt(Instant.now());

        PredictionRecord record2 = new PredictionRecord();
        record2.setId(2L);
        record2.setAge(30);
        record2.setGender("female");
        record2.setPhq1(2);
        record2.setPhq2(2);
        record2.setPhq3(2);
        record2.setPhq4(2);
        record2.setPhq5(2);
        record2.setPhq6(2);
        record2.setPhq7(2);
        record2.setPhq8(2);
        record2.setPhq9(2);
        record2.setPrediction(0);
        record2.setProbability(0.88);
        record2.setThreshold(0.30);
        record2.setModel("logreg");
        record2.setCreatedAt(Instant.now());

        List<PredictionRecord> mockRecords = List.of(record1, record2);

        Mockito.when(repo.findAll(Mockito.any(Sort.class))).thenReturn(mockRecords);

        mockMvc.perform(get("/api/predictions"))
              .andExpect(status().isOk())
              .andExpect(jsonPath("$.length()").value(2))
              .andExpect(jsonPath("$[0].age").value(25))
              .andExpect(jsonPath("$[1].gender").value("female"));
    }

}
