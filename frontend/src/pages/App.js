// src/pages/App.js
import React, { useState } from "react";
import PHQForm from "../components/PHQForm";
import ShapChart from "../components/ShapChart";
import { sendPrediction, sendFeedback, getExplanation } from "../services/api";

const App = () => {
  const [result, setResult] = useState(null);
  const [feedback, setFeedback] = useState("");
  const [shapData, setShapData] = useState(null);

  const handlePredict = async (formData) => {
    try {
      const res = await sendPrediction(formData);
      setResult(res.data);

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
                <p><strong>Model:</strong> {result.model}</p>

                {/* SHAP Explanation */}
                {shapData && (
                  <>
                    <h5 className="mt-4">Top Contributing Features</h5>
                    <ShapChart data={shapData} />
                  </>
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
