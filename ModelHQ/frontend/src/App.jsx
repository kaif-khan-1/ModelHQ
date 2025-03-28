import React, { useEffect } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar/Navbar";
import Home from "./pages/Home/Home";
import Model from "./pages/Models/Models";
import Stock_prediction from "./pages/Stock Prediction Model/Stock_prediction";
import Spam from './pages/Spam detection/Spam_detection'
import Movie from './pages/Movie Recommendation/Movie_recommendation'
import Housing from './pages/Housing price prediction/hpp'
import Heart_disease from './pages/Heart disease/Heart_disease'
import Diabetes_prediction from './pages/Diabetes_prediction/diabetes_prediction'
import BreastCancer from "./pages/Breast Cancer/BreastCancer";
import Lenis from "@studio-freight/lenis";

const App = () => {
  useEffect(() => {
    const lenis = new Lenis({
      smooth: true,
      lerp: 0.05,
      infinite: false,
      speed: 0.5,
    });

    const raf = (time) => {
      lenis.raf(time);
      requestAnimationFrame(raf);
    };

    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);

  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<Model />} />
        <Route path='/stock_prediction' element={<Stock_prediction/>} />
        <Route path='/spam' element={<Spam/>} />
        <Route path='/movie' element={<Movie/>} />
        <Route path='/hpp' element={<Housing/>} />
        <Route path='/heart' element={<Heart_disease/>} />
        <Route path='/diabetes' element={<Diabetes_prediction/>} />
        <Route path='/BreastCancer' element={<BreastCancer/>} />
      </Routes>
    </Router>
  );
};

export default App;
