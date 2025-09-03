import React, { useState } from "react";

const PHQForm = ({ onSubmit }) => {
  const [inputs, setInputs] = useState({
    age: "",
    gender: "",
    phq: Array(9).fill(""),
  });

  const handleChange = (e, index = null) => {
    if (index !== null) {
      const updated = [...inputs.phq];
      updated[index] = e.target.value;
      setInputs({ ...inputs, phq: updated });
    } else {
      setInputs({ ...inputs, [e.target.name]: e.target.value });
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (inputs.age < 10 || inputs.age > 120) {
      alert("Age must be between 10 and 120.");
      return;
    }

    const payload = {
      age: parseInt(inputs.age),
      gender: inputs.gender,
      phq1: parseInt(inputs.phq[0]),
      phq2: parseInt(inputs.phq[1]),
      phq3: parseInt(inputs.phq[2]),
      phq4: parseInt(inputs.phq[3]),
      phq5: parseInt(inputs.phq[4]),
      phq6: parseInt(inputs.phq[5]),
      phq7: parseInt(inputs.phq[6]),
      phq8: parseInt(inputs.phq[7]),
      phq9: parseInt(inputs.phq[8]),
    };

    onSubmit(payload);
  };

  return (
    <form onSubmit={handleSubmit}>
      <div className="mb-3">
        <label>Age:</label>
        <input type="number" name="age" className="form-control" required min="10" max="120" value={inputs.age} onChange={handleChange} />
      </div>

      <div className="mb-3">
        <label>Gender:</label>
        <select name="gender" className="form-control" required value={inputs.gender} onChange={handleChange}>
          <option value="">-- Select --</option>
          <option value="male">Male</option>
          <option value="female">Female</option>
        </select>
      </div>

      <label>PHQ-9 Scores:</label>
      {inputs.phq.map((val, i) => (
        <input
          key={i}
          type="number"
          min="0"
          max="3"
          className="form-control my-1"
          placeholder={`PHQ${i + 1}`}
          value={val}
          onChange={(e) => handleChange(e, i)}
          required
        />
      ))}

      <button type="submit" className="btn btn-primary mt-3">Predict</button>
    </form>
  );
};

export default PHQForm;
