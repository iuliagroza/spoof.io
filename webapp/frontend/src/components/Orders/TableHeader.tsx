import React from 'react';
import './TableHeader.scss';

interface TableHeaderProps {
    type: string;
}

const TableHeader: React.FC<TableHeaderProps> = ({ type }) => {
    const handleMouseEnter = (event: React.MouseEvent<HTMLElement>) => {
        const target = event.target as HTMLElement;
        if (target.scrollWidth > target.clientWidth) {
            target.title = target.innerText;
        }
    };

    return (
        <div className="table-header">
            <table>
                <thead>
                    <tr>
                        {type === "spoofing" ? (
                            <>
                                <th onMouseEnter={handleMouseEnter}>Detected Spoofed<br />Order ID</th>
                                <th onMouseEnter={handleMouseEnter}>Time</th>
                                <th onMouseEnter={handleMouseEnter}>Anomaly<br />Score</th>
                                <th onMouseEnter={handleMouseEnter}>Spoofing<br />Threshold</th>
                            </>
                        ) : (
                            <>
                                <th onMouseEnter={handleMouseEnter}>Order ID</th>
                                <th onMouseEnter={handleMouseEnter}>Type</th>
                                <th onMouseEnter={handleMouseEnter}>Size</th>
                                <th onMouseEnter={handleMouseEnter}>Price</th>
                                <th onMouseEnter={handleMouseEnter}>Time</th>
                                <th onMouseEnter={handleMouseEnter}>Reason</th>
                                <th onMouseEnter={handleMouseEnter}>Remaining<br />Size</th>
                                <th onMouseEnter={handleMouseEnter}>Trade ID</th>
                            </>
                        )}
                    </tr>
                </thead>
            </table>
        </div>
    );
};

export default TableHeader;