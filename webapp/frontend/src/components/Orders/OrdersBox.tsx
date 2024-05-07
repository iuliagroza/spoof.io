import React from 'react';
import './OrdersBox.scss';
import { Order } from '../../types/order';

interface OrdersBoxProps {
    orders: Order[];
}

const OrdersBox: React.FC<OrdersBoxProps> = ({ orders }) => {
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
                    </tr>
                </thead>
                <tbody>
                    {orders.map(order => (
                        <tr key={order.id}>
                            <td>{order.id}</td>
                            <td>{order.type}</td>
                            <td>{order.size}</td>
                            <td>{order.price}</td>
                            <td>{order.status}</td>
                            <td>{order.time}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default OrdersBox;
