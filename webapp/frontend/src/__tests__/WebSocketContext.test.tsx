import React from 'react';
import { render, waitFor, act } from '@testing-library/react';
import { WebSocketProvider, useWebSocket } from '../websocket/WebSocketContext';
import { Order } from '../types/order';

jest.spyOn(console, 'error').mockImplementation(() => { });

class WebSocketMock {
    static instances: WebSocketMock[] = [];
    onopen: (() => void) | null = null;
    onmessage: ((event: { data: string }) => void) | null = null;
    onclose: (() => void) | null = null;
    onerror: ((event: any) => void) | null = null;

    constructor(url: string) {
        WebSocketMock.instances.push(this);
    }

    send() { }
    close() {
        if (this.onclose) {
            this.onclose();
        }
    }
}

global.WebSocket = WebSocketMock as any;

const TestComponent: React.FC = () => {
    const { regularOrders, spoofingOrders } = useWebSocket();

    return (
        <div>
            <div>
                Regular Orders:
                {regularOrders.map((order: Order, index: number) => (
                    <div key={index} data-testid="regular-order">
                        {order.order_id}
                    </div>
                ))}
            </div>
            <div>
                Spoofing Orders:
                {spoofingOrders.map((order: Order, index: number) => (
                    <div key={index} data-testid="spoofing-order">
                        {order.order_id}
                    </div>
                ))}
            </div>
        </div>
    );
};

describe('WebSocketContext', () => {
    afterEach(() => {
        WebSocketMock.instances = [];
        jest.clearAllMocks();
    });

    test('connects to WebSocket and receives messages', async () => {
        const regularOrder = { order_id: 'regular_1', time: '2023-05-01T12:00:00Z' };
        const spoofingOrder = { order_id: 'spoofing_1', time: '2023-05-01T12:00:00Z', is_spoofing: true };

        const { getAllByTestId } = render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        await act(async () => {
            WebSocketMock.instances[0].onopen?.();
        });

        await act(async () => {
            WebSocketMock.instances[0].onmessage?.({
                data: JSON.stringify(regularOrder),
            });
        });

        await act(async () => {
            WebSocketMock.instances[0].onmessage?.({
                data: JSON.stringify(spoofingOrder),
            });
        });

        await waitFor(() => {
            const regularOrders = getAllByTestId('regular-order');
            const spoofingOrders = getAllByTestId('spoofing-order');

            expect(regularOrders).toHaveLength(1);
            expect(regularOrders[0]).toHaveTextContent('regular_1');

            expect(spoofingOrders).toHaveLength(1);
            expect(spoofingOrders[0]).toHaveTextContent('spoofing_1');
        });
    });

    test('handles WebSocket errors gracefully', async () => {
        const consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation(() => { });

        render(
            <WebSocketProvider>
                <TestComponent />
            </WebSocketProvider>
        );

        await act(async () => {
            WebSocketMock.instances[0].onerror?.({ message: 'WebSocket error' });
        });

        await waitFor(() => {
            expect(consoleErrorSpy).toHaveBeenCalledWith('WebSocket Error:', { message: 'WebSocket error' });
        });

        consoleErrorSpy.mockRestore();
    });
});
