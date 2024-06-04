import React, { useState } from 'react';
import './OrdersBox.scss';
import { useWebSocket } from '../../websocket/WebSocketContext';
import TableHeader from './TableHeader';

interface OrdersBoxProps {
    type: string;
    className?: string;
}

const getSeverityClass = (anomalyScore: number, spoofingThreshold: number) => {
    const difference = Math.abs(anomalyScore - spoofingThreshold);

    if (difference < 5) {
        return 'severity-low';
    } else if (difference < 10) {
        return 'severity-medium';
    } else {
        return 'severity-high';
    }
};

const OrdersBox: React.FC<OrdersBoxProps> = ({ type, className }) => {
    const { regularOrders, spoofingOrders } = useWebSocket();
    const orders = type === "spoofing" ? spoofingOrders : regularOrders;

    const [copied, setCopied] = useState(false);
    const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });

    const handleMouseEnter = (event: React.MouseEvent<HTMLElement>) => {
        const target = event.target as HTMLElement;
        if (target.scrollWidth > target.clientWidth) {
            target.title = target.innerText;
        }
    };

    const handleRowClick = (event: React.MouseEvent<HTMLTableRowElement>) => {
        const target = event.currentTarget as HTMLTableRowElement;
        const cells = Array.from(target.querySelectorAll('td'));
        const rowText = cells.map(cell => cell.innerText).join('; ');

        setPopupPosition({ x: event.clientX, y: event.clientY });

        navigator.clipboard.writeText(rowText).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 2000); // Show message for 2 seconds
        }).catch(err => {
            console.error('Failed to copy:', err);
        });
    };

    return (
        <div className={`orders-box ${className || ''}`}>
            <TableHeader type={type} />
            <div className="orders-body">
                <table className="orders-table">
                    <tbody>
                        {orders.map((order, index) => {
                            const anomalyScore = (order.anomaly_score ?? 0) * 100;
                            const spoofingThreshold = (order.spoofing_threshold ?? 0) * 100;
                            const severityClass = type === "spoofing" ? getSeverityClass(anomalyScore, spoofingThreshold) : '';

                            return (
                                <tr key={index} className={severityClass} onClick={handleRowClick}>
                                    <td onMouseEnter={handleMouseEnter}>{order.order_id ?? 'N/A'}</td>
                                    {type === "spoofing" ? (
                                        <>
                                            <td onMouseEnter={handleMouseEnter}>{order.time}</td>
                                            <td onMouseEnter={handleMouseEnter}>{anomalyScore.toFixed(2)}%</td>
                                            <td onMouseEnter={handleMouseEnter}>{spoofingThreshold.toFixed(2)}%</td>
                                        </>
                                    ) : (
                                        <>
                                            <td onMouseEnter={handleMouseEnter}>{order.type ?? 'N/A'}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.size ?? 0}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.price ?? 0}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.time}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.reason ?? 'N/A'}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.remaining_size ?? 0}</td>
                                            <td onMouseEnter={handleMouseEnter}>{order.trade_id ?? 'N/A'}</td>
                                        </>
                                    )}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
                {copied && (
                    <div
                        className="copied-popup"
                        style={{ top: `${popupPosition.y}px`, left: `${popupPosition.x}px` }}
                    >
                        Copied to clipboard
                    </div>
                )}
            </div>
        </div>
    );
};

export default OrdersBox;
