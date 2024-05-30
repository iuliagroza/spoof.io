import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import Navbar from '../components/Navbar/Navbar';

test('renders Navbar component', () => {
    render(
        <BrowserRouter>
            <Navbar setNavbarHeight={() => { }} />
        </BrowserRouter>
    );

    expect(screen.getByText(/spoof.io/i)).toBeInTheDocument();
    expect(screen.getByText(/Detector/i)).toBeInTheDocument();
    expect(screen.getByText(/Training Logs/i)).toBeInTheDocument();
    expect(screen.getByText(/Hypertuning Logs/i)).toBeInTheDocument();
});
