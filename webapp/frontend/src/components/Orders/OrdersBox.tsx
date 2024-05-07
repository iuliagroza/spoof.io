import React from 'react';
import './OrdersBox.scss';

interface OrdersBoxProps {
    orders: string[];
}

const OrdersBox: React.FC<OrdersBoxProps> = ({ orders }) => {
    return (
        <div className="orders-box">
            <ul>
                {orders.map((order, index) => (
                    <li key={index}>{order}</li>
                ))}
            </ul>
        </div>
    );
};

export default OrdersBox;
