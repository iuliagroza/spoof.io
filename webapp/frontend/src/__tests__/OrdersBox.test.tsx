import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import OrdersBox from '../components/Orders/OrdersBox';

// Mock the useWebSocket hook
jest.mock('../WebSocketContext', () => ({
    useWebSocket: () => ({
        regularOrders: [
            {
                order_id: '1234',
                type: 'buy',
                size: 10,
                price: 100,
                time: '2024-05-29T12:00:00Z',
                reason: 'strategy',
                remaining_size: 5,
                trade_id: 'abcd'
            }
        ],
        spoofingOrders: [
            {
                order_id: '5678',
                time: '2024-05-29T12:01:00Z',
                anomaly_score: 0.8885954762677892,
                spoofing_threshold: 0.8628708924383861
            }
        ]
    })
}));

describe('OrdersBox Component', () => {
    it('renders regular orders correctly', () => {
        render(<OrdersBox type="regular" />);

        expect(screen.getByText('Order ID')).toBeInTheDocument();
        expect(screen.getByText('1234')).toBeInTheDocument();
        expect(screen.getByText('buy')).toBeInTheDocument();
        expect(screen.getByText('10')).toBeInTheDocument();
        expect(screen.getByText('100')).toBeInTheDocument();
        expect(screen.getByText('2024-05-29T12:00:00Z')).toBeInTheDocument();
        expect(screen.getByText('strategy')).toBeInTheDocument();
        expect(screen.getByText('5')).toBeInTheDocument();
        expect(screen.getByText('abcd')).toBeInTheDocument();
    });

    it('renders spoofing orders correctly', () => {
        render(<OrdersBox type="spoofing" />);

        expect(screen.getByText('Detected Spoofed Order ID')).toBeInTheDocument();
        expect(screen.getByText('5678')).toBeInTheDocument();
        expect(screen.getByText('2024-05-29T12:01:00Z')).toBeInTheDocument();
        expect(screen.getByText('88.86%')).toBeInTheDocument();
        expect(screen.getByText('86.29%')).toBeInTheDocument();
    });
});
