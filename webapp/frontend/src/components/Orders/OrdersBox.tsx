import React from 'react';
import './OrdersBox.scss';
import { useWebSocket } from '../../WebSocketContext';

const OrdersBox: React.FC = () => {
    const { orders } = useWebSocket();

    return (
        <div className="orders-box">
            <table>
                <thead>
                    <tr>
                        <th>Order ID</th>
                        <th>Type</th>
                        <th>Size</th>
                        <th>Price</th>
                        <th>Status</th>
                        <th>Time</th>
                        <th>Reason</th>
                        <th>Remaining Size</th>
                        <th>Trade ID</th>
                    </tr>
                </thead>
                <tbody>
                    {orders.map(order => (
                        <tr key={order.order_id}>
                            <td>{order.order_id}</td>
                            <td>{order.type}</td>
                            <td>{order.size}</td>
                            <td>{order.price}</td>
                            <td>{order.status}</td>
                            <td>{order.time}</td>
                            <td>{order.reason}</td>
                            <td>{order.remaining_size}</td>
                            <td>{order.trade_id}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default OrdersBox;
