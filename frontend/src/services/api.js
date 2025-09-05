import axios from "axios";

const API_BASE = "http://localhost:8080/api";
const ML_BASE  = "http://localhost:8000";  

export const sendPrediction = (data) =>
  axios.post(`${API_BASE}/predict`, data);

export const sendFeedback = (feedbackData) =>
  axios.post(`${API_BASE}/feedback`, feedbackData);

export const getExplanation = (data) =>
  axios.post(`${API_BASE}/explain`, data);

export async function getModelMetrics(model) {
  const res = await axios.get(`${ML_BASE}/metrics`, { params: { model } });
  return res.data;
}
