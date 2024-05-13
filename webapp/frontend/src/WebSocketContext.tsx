import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import { Order } from './types/order';  // Ensure this type is defined in your project according to your needs

interface WebSocketContextType {
    regularOrders: Order[];
    spoofingOrders: Order[];
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined);

interface WebSocketProviderProps {
    children: ReactNode;
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
    const [regularOrders, setRegularOrders] = useState<Order[]>([]);
    const [spoofingOrders, setSpoofingOrders] = useState<Order[]>([]);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/orders/');

        ws.onopen = () => {
            console.log('WebSocket Connected');
        };

        ws.onmessage = (event) => {
            // console.log("Received message:", event.data);
            try {
                const data = JSON.parse(event.data);
                if (data.is_spoofing) {
                    console.log(data);
                    setSpoofingOrders(prev => [...prev, data]);
                } else {
                    setRegularOrders(prev => [...prev, data]);
                }
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
            if (ws.readyState === WebSocket.OPEN) {
                console.log('Closing WebSocket');
                ws.close();
            }
        };
    }, []);

    return (
        <WebSocketContext.Provider value={{ regularOrders, spoofingOrders }}>
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