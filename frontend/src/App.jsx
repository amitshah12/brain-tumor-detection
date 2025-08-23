import React, { useState } from "react";
import Imageinput from "./components/Imageinput";
import Prediction from "./components/Prediction";

function App() {
  const [prediction, setPrediction] = useState(null);

  return (
    <div className="main-container">
      <div className="left-panel">
        <Imageinput setPrediction={setPrediction} />
      </div>

      <div className="right-panel">
        <Prediction prediction={prediction} />
      </div>
    </div>
  );
}

export default App;