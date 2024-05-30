import { render, screen } from '@testing-library/react';
import Hero from '../components/Hero/Hero';

test('renders Hero component', () => {
    render(<Hero setHeroHeight={() => { }} />);

    expect(screen.getByText(/Secure Your LUNA Transactions Against Spoofing/i)).toBeInTheDocument();
    expect(screen.getByText(/Protect yourself from market manipulations/i)).toBeInTheDocument();
});
