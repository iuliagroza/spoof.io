import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders navbar and hero section', () => {
  render(<App />);
  const logoElement = screen.getByText(/spoof.io/i);
  expect(logoElement).toBeInTheDocument();

  const heroTitleElement = screen.getByText(/secure your luna transactions against spoofing/i);
  expect(heroTitleElement).toBeInTheDocument();

  const heroDescriptionElement = screen.getByText(/protect yourself from market manipulations with real-time monitoring of spoofing & layering attempts on LUNA/i);
  expect(heroDescriptionElement).toBeInTheDocument();
});