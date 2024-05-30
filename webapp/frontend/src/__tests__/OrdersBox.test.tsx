import React from 'react';
import { render, screen } from '@testing-library/react';
import OrdersBox from '../components/Orders/OrdersBox';
import { WebSocketProvider } from '../websocket/WebSocketContext';

const mockWebSocketData = {
    regularOrders: [
        { order_id: '123', type: 'buy', size: 10, price: 100, time: '10:00', reason: 'reason', remaining_size: 5, trade_id: 'trade123' }
    ],
    spoofingOrders: [
        { order_id: '456', time: '11:00', anomaly_score: 0.9, spoofing_threshold: 0.85 }
    ]
};

const renderWithProvider = (ui: React.ReactElement) => {
    return render(
        <WebSocketProvider>
            {ui}
        </WebSocketProvider>
    );
};

test('renders OrdersBox component', () => {
    renderWithProvider(<OrdersBox type="regular" />);

    expect(screen.getByText(/Order ID/i)).toBeInTheDocument();
    expect(screen.getByText(/Type/i)).toBeInTheDocument();
    const sizeElements = screen.getAllByText(/Size/i);
    expect(sizeElements.length).toBeGreaterThan(0);
});

test('renders spoofing OrdersBox component', () => {
    renderWithProvider(<OrdersBox type="spoofing" />);

    expect(screen.getByText(/Detected Spoofed/i)).toBeInTheDocument();
    expect(screen.getByText(/Time/i)).toBeInTheDocument();

    const matchAnomalyScore = (_: string, element: HTMLElement | null): boolean => {
        if (element) {
            const hasText = (text: string) => element.textContent?.includes(text) ?? false;
            return hasText("Anomaly") && hasText("Score");
        }
        return false;
    };

    const matchSpoofingThreshold = (_: string, element: HTMLElement | null): boolean => {
        if (element) {
            const hasText = (text: string) => element.textContent?.includes(text) ?? false;
            return hasText("Spoofing") && hasText("Threshold");
        }
        return false;
    };

    expect(screen.getAllByText((_, element) => matchAnomalyScore(_, element as HTMLElement)).length).toBeGreaterThan(0);
    expect(screen.getAllByText((_, element) => matchSpoofingThreshold(_, element as HTMLElement)).length).toBeGreaterThan(0);
});
