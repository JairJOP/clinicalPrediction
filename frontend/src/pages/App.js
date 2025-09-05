import React, { useEffect, useState } from "react";
import PHQForm from "../components/PHQForm";
import ShapChart from "../components/ShapChart";
import { sendPrediction, sendFeedback, getExplanation, getModelMetrics} from "../services/api";

const App = () => {
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [shapData, setShapData] = useState(null);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");

  useEffect(() => {
    if (selectedModel) {
      getModelMetrics(selectedModel)
        .then((data) => setModelMetrics(data))
        .catch((err) => console.error("Failed to load model metrics", err));
    }
  }, [selectedModel]);

  const handlePredict = async (formData) => {
    try {
      const res = await sendPrediction(formData);
      setResult(res.data);
      setSelectedModel(res.data.model);

      // Fetch SHAP explanation
      const shapRes = await getExplanation(formData);
      setShapData(shapRes.data.top_features); // expected shape from backend

    } catch (error) {
      console.error("Prediction or explanation failed", error);
      alert("Prediction failed. Check backend/Flask server.");
    }
  };

  const handleFeedback = async () => {
    try {
      await sendFeedback({
        predictionId: result?.id,
        message: feedback,
      });
      alert("Feedback submitted.");
      setFeedback("");
    } catch (error) {
      console.error("Feedback error", error);
      alert("Error submitting feedback");
    }
  };

  const modelLabels = {
    logreg: "Logistic Regression",
    rf: "Random Forest",
    xgb: "XGBoost"
  };

  return (
    <div className="container mt-5">
      <div className="row justify-content-center">
        <div className="col-md-8">

          {/* Form Card */}
          <div className="card shadow-sm border-0 mb-4">
            <div className="card-body">
              <h3 className="card-title mb-4">Clinical Prediction Tool</h3>
              <PHQForm onSubmit={handlePredict} />
            </div>
          </div>

          {/* Prediction Results Card */}
          {result && (
            <div className="card shadow-sm border-0 mb-4">
              <div className="card-body">
                <h4 className="card-title">Prediction Result</h4>
                <p><strong>Outcome:</strong> {result.prediction === 1 ? "Depression" : "No Depression"}</p>
                <p><strong>Confidence:</strong> {(result.probability * 100).toFixed(2)}%</p>
                <p><strong>Model:</strong> {modelLabels[result.model] || result.model}</p>

                {/* SHAP Explanation */}
                {shapData && (
                  <>
                    <h5 className="mt-4">Top Contributing Features</h5>
                    <ShapChart data={shapData} />
                  </>
                )}

                {modelMetrics && modelMetrics.test_tuned && modelMetrics.cv && (
                <div className="mt-4">
                  <h5>Model Evaluation Metrics</h5>

                  <p><strong>Test Accuracy:</strong> {modelMetrics.test_tuned.accuracy.toFixed(3)}</p>
                  <p><strong>Precision:</strong> {modelMetrics.test_tuned.precision.toFixed(3)}</p>
                  <p><strong>Recall:</strong> {modelMetrics.test_tuned.recall.toFixed(3)}</p>
                  <p><strong>F1 Score:</strong> {modelMetrics.test_tuned.f1.toFixed(3)}</p>
                  <p><strong>ROC AUC:</strong> {modelMetrics.test_tuned.roc_auc.toFixed(3)}</p>

                  <p className="mt-3">
                    <strong>CV AUC:</strong> {modelMetrics.cv.roc_auc_mean.toFixed(3)} ± {modelMetrics.cv.roc_auc_std.toFixed(3)}
                  </p>
                </div>
              )}

                {/* Feedback Section */}
                <div className="mt-4">
                  <label className="form-label">Flag / Feedback:</label>
                  <textarea
                    className="form-control"
                    rows="3"
                    placeholder="Why do you think this prediction is wrong or misleading?"
                    value={feedback}
                    onChange={(e) => setFeedback(e.target.value)}
                  />
                  <button className="btn btn-warning mt-2" onClick={handleFeedback}>
                    Submit Feedback
                  </button>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>
    </div>
  );
};

export default App;
