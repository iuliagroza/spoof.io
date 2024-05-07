import React from 'react';
import './App.scss';
import Hero from './components/Hero/Hero';
import Navbar from './components/Navbar/Navbar';

const App: React.FC = () => {
  return (
    <div className="App">
      <Navbar />
      <Hero />
    </div>
  );
}

export default App;
