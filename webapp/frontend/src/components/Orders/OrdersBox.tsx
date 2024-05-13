import React from 'react';
import './OrdersBox.scss';
import { useWebSocket } from '../../WebSocketContext';

const OrdersBox: React.FC<{ type: string }> = ({ type }) => {
    const { regularOrders, spoofingOrders } = useWebSocket();
    const orders = type === "spoofing" ? spoofingOrders : regularOrders;

    return (
        <div className="orders-box">
            <table>
                <thead>
                    <tr>
                        {type === "spoofing" ? (
                            <>
                                <th>Order ID</th>
                                <th>Time</th>
                                <th>Anomaly Score</th>
                                <th>Spoofing Threshold</th>
                            </>
                        ) : (
                            <>
                                <th>Order ID</th>
                                <th>Type</th>
                                <th>Size</th>
                                <th>Price</th>
                                <th>Time</th>
                                <th>Reason</th>
                                <th>Remaining Size</th>
                                <th>Trade ID</th>
                            </>
                        )}
                    </tr>
                </thead>
                <tbody>
                    {orders.map((order, index) => (
                        <tr key={index}>
                            <td>{order.order_id}</td>
                            {type === "spoofing" ? (
                                <>
                                    <td>{order.time}</td>
                                    <td>{order.anomaly_score}</td>
                                    <td>{order.spoofing_threshold}</td>
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
    );
};

export default OrdersBox;