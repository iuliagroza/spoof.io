import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { Order } from './types/order';  // Ensure this type is defined in your project according to your needs

interface WebSocketContextType {
    orders: Order[];
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
    children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
    const [orders, setOrders] = useState<Order[]>([]);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/test/');

        ws.onopen = () => {
            console.log('WebSocket Connected');
        };

        ws.onmessage = (event) => {
            try {
                const order: Order = JSON.parse(event.data);
                setOrders(prevOrders => [...prevOrders, order]);
            } catch (error) {
                console.error('Failed to parse order from WebSocket message:', error);
            }
        };

        ws.onclose = () => {
            console.log('WebSocket Disconnected');
        };

        ws.onerror = (error) => {
            console.log('WebSocket Error:', error);
        };

        return () => {
            if (ws.readyState == 1) {
                console.log('Closing WebSocket');
                ws.close();
            }
        };
    }, []);

    return (
        <WebSocketContext.Provider value={{ orders }}>
            {children}
        </WebSocketContext.Provider>
    );
};

export const useWebSocket = (): WebSocketContextType => {
    const context = useContext(WebSocketContext);
    if (!context) {
        throw new Error('useWebSocket must be used within a WebSocketProvider');
    }
    return context;
};