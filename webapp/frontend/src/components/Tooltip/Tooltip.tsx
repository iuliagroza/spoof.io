import React from 'react';
import './Tooltip.scss';

interface TooltipProps {
    content: string;
    children: React.ReactNode;
}

const Tooltip: React.FC<TooltipProps> = ({ content, children }) => {
    return (
        <div className="tooltip-container">
            {children}
            <div className="tooltip-text">{content}</div>
        </div>
    );
};

export default Tooltip;
