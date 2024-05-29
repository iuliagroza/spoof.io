import React from 'react';
import './OrdersBox.scss';
import { useWebSocket } from '../../WebSocketContext';
import TableHeader from './TableHeader';

interface OrdersBoxProps {
    type: string;
    className?: string;
}

const OrdersBox: React.FC<OrdersBoxProps> = ({ type, className }) => {
    const { regularOrders, spoofingOrders } = useWebSocket();
    const orders = type === "spoofing" ? spoofingOrders : regularOrders;

    return (
        <div className={`orders-box ${className || ''}`}>
            <TableHeader type={type} />
            <div className="orders-body">
                <table className="orders-table">
                    <tbody>
                        {orders.map((order, index) => (
                            <tr key={index}>
                                <td>{order.order_id ?? 'N/A'}</td>
                                {type === "spoofing" ? (
                                    <>
                                        <td>{order.time}</td>
                                        <td>{((order.anomaly_score ?? 0) * 100).toFixed(2)}%</td>
                                        <td>{((order.spoofing_threshold ?? 0) * 100).toFixed(2)}%</td>
                                    </>
                                ) : (
                                    <>
                                        <td>{order.type ?? 'N/A'}</td>
                                        <td>{order.size ?? 0}</td>
                                        <td>{order.price ?? 0}</td>
                                        <td>{order.time}</td>
                                        <td>{order.reason ?? 'N/A'}</td>
                                        <td>{order.remaining_size ?? 0}</td>
                                        <td>{order.trade_id ?? 'N/A'}</td>
                                    </>
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default OrdersBox;
