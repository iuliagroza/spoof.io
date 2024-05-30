import { render, screen } from '@testing-library/react';
import TableHeader from '../components/Orders/TableHeader';

test('renders regular TableHeader', () => {
    render(<TableHeader type="regular" />);

    expect(screen.getByText(/Order ID/i)).toBeInTheDocument();
    expect(screen.getByText(/Type/i)).toBeInTheDocument();
    const sizeElements = screen.getAllByText(/Size/i);
    expect(sizeElements.length).toBeGreaterThan(0);
});

test('renders spoofing TableHeader', () => {
    render(<TableHeader type="spoofing" />);

    const matchDetectedSpoofedOrderID = (content: string, element: Element | null): boolean => {
        if (element instanceof HTMLElement) {
            const hasText = (text: string) => element.textContent?.includes(text) ?? false;
            return hasText('Detected Spoofed') && hasText('Order ID');
        }
        return false;
    };

    expect(screen.getAllByText(matchDetectedSpoofedOrderID).length).toBeGreaterThan(0);

    expect(screen.getByText(/Time/i)).toBeInTheDocument();

    const matchAnomalyScore = (content: string, element: Element | null): boolean => {
        if (element instanceof HTMLElement) {
            const hasText = (text: string) => element.textContent?.includes(text) ?? false;
            return hasText('Anomaly') && hasText('Score');
        }
        return false;
    };

    const matchSpoofingThreshold = (content: string, element: Element | null): boolean => {
        if (element instanceof HTMLElement) {
            const hasText = (text: string) => element.textContent?.includes(text) ?? false;
            return hasText('Spoofing') && hasText('Threshold');
        }
        return false;
    };

    expect(screen.getAllByText(matchAnomalyScore).length).toBeGreaterThan(0);
    expect(screen.getAllByText(matchSpoofingThreshold).length).toBeGreaterThan(0);
});
