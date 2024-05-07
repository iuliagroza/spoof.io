import React from 'react';
import './App.scss';
import Hero from './components/Hero/Hero';
import Navbar from './components/Navbar/Navbar';
import OrdersBox from './components/Orders/OrdersBox';

const App: React.FC = () => {
  const orders = ["Order 1", "Order 2", "Order 3"];

  return (
    <div className="App">
      <Navbar />
      <Hero />
      <div className="app-container">
        <OrdersBox orders={orders} />
        <OrdersBox orders={orders} />
      </div>
    </div>
  );
}

export default App;
