import React from 'react';
import './TableHeader.scss';

interface TableHeaderProps {
    type: string;
}

const TableHeader: React.FC<TableHeaderProps> = ({ type }) => {
    return (
        <div className="table-header">
            <table>
                <thead>
                    <tr>
                        {type === "spoofing" ? (
                            <>
                                <th>Detected Spoofed Order ID</th>
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
            </table>
        </div>
    );
};

export default TableHeader;
