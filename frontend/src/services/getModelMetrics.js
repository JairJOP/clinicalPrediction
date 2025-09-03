import axios from "axios";

const BASE_URL = "hhtp://127.0.0.1:8000";

export const getModelMetrics = (model) =>
  axios.get(`$BASE_URL}/metrics?model=${model}`);