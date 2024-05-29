import React, { useState } from 'react';
import './App.scss';
import Hero from './components/Hero/Hero';
import Navbar from './components/Navbar/Navbar';
import OrdersBox from './components/Orders/OrdersBox';
import { WebSocketProvider } from './WebSocketContext';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import TrainingLogs from './logs_routes/TrainingLogs';
import HypertuningLogs from './logs_routes/HypertuningLogs';

const App: React.FC = () => {
  const [navbarHeight, setNavbarHeight] = useState(0);
  const [heroHeight, setHeroHeight] = useState(0);

  const availableHeight = `calc(85vh - ${navbarHeight}px - ${heroHeight}px)`;

  return (
    <WebSocketProvider>
      <Router>
        <div className="App">
          <Navbar setNavbarHeight={setNavbarHeight} />
          <Routes>
            <Route path="/" element={<>
              <Hero setHeroHeight={setHeroHeight} />
              <div className="app-container" style={{ height: availableHeight }}>
                <OrdersBox type="regular" className="orders-box-regular" />
                <OrdersBox type="spoofing" className="orders-box-spoofing" />
              </div>
            </>} />
            <Route path="/training-logs" element={<TrainingLogs />} />
            <Route path="/hypertuning-logs" element={<HypertuningLogs />} />
          </Routes>
        </div>
      </Router>
    </WebSocketProvider>
  );
}

export default App;
