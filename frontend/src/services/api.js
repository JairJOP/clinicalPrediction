import axios from "axios";

const API_BASE = "http://localhost:8080/api";

export const sendPrediction = (data) =>
  axios.post(`${API_BASE}/predict`, data);

export const sendFeedback = (feedbackData) =>
  axios.post(`${API_BASE}/feedback`, feedbackData);

export const getExplanation = (data) =>
  axios.post(`${API_BASE}/explain`, data);

export async function getModelMetrics(model) {
  const response = await fetch(`http://127.0.0.1:8000/metrics?model=${model}`);
  if (!response.ok) {
    throw new Error("Failed to fetch model metrics");
  }
  return await response.json();
}
