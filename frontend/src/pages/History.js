import React, { useEffect, useState } from 'react';
import axios from 'axios';

export default function History() {
  const [records, setRecords] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8080/api/predictions')
      .then((res) => setRecords(res.data))
      .catch((err) => alert("Error: " + err.message));
  }, []);

  return (
    <div style={{ padding: 20 }}>
      <h2>Prediction History</h2>
      <ul>
        {records.map((rec, i) => (
          <li key={i}>
            Age: {rec.age}, Gender: {rec.gender}, Prediction: {rec.prediction}, Prob: {rec.probability}
          </li>
        ))}
      </ul>
    </div>
  );
}
