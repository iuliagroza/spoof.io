import { render } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders navbar and hero', () => {
  const { container } = render(<App />);
  expect(container.querySelector('.navbar')).toBeInTheDocument();
  expect(container.querySelector('.hero')).toBeInTheDocument();
});
