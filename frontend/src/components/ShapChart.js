import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const ShapChart = ({ data }) => (
  <ResponsiveContainer width="100%" height={300}>
    <BarChart layout="vertical" data={data}>
      <XAxis type="number" />
      <YAxis type="category" dataKey="feature" />
      <Tooltip />
      <Bar dataKey="shap_value" fill="#8884d8" />
    </BarChart>
  </ResponsiveContainer>
);

export default ShapChart;
