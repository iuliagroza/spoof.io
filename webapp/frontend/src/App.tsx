import React from 'react';
import './App.scss';
import Hero from './components/Hero/Hero';
import Navbar from './components/Navbar/Navbar';
import OrdersBox from './components/Orders/OrdersBox';
import { Order } from './types/order';

const mockOrders: Order[] = [
  {
    id: '1',
    type: 'Buy',
    size: 50,
    price: 3000,
    status: 'Completed',
    time: '2022-05-11T19:04:37Z'
  },
  {
    id: '2',
    type: 'Sell',
    size: 30,
    price: 3050,
    status: 'Pending',
    time: '2022-05-11T20:15:42Z'
  },
  {
    id: '3',
    type: 'Buy',
    size: 20,
    price: 2980,
    status: 'Canceled',
    time: '2022-05-11T21:07:23Z'
  },
  {
    id: '4',
    type: 'Sell',
    size: 40,
    price: 3100,
    status: 'Completed',
    time: '2022-05-11T22:30:00Z'
  }
];

const App: React.FC = () => {
  return (
    <div className="App">
      <Navbar />
      <Hero />
      <div className="app-container">
        <OrdersBox orders={mockOrders} />
        <OrdersBox orders={mockOrders} />
      </div>
    </div>
  );
}

export default App;
