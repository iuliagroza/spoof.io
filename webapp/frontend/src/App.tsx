import React from 'react';
import './App.scss';
import Hero from './components/Hero/Hero';
import Navbar from './components/Navbar/Navbar';
import OrdersBox from './components/Orders/OrdersBox';
import { WebSocketProvider } from './WebSocketContext';


const App: React.FC = () => {
  return (
    <WebSocketProvider>
      <div className="App">
        <Navbar />
        <Hero />
        <div className="app-container">
          <OrdersBox />
        </div>
      </div>
    </WebSocketProvider>
  );
}

export default App;
