import React from 'react';

const HypertuningLogs: React.FC = () => {
    return (
        <iframe
            src="/hypertuning_logs.html"
            style={{ width: '100%', height: '100vh', border: 'none' }}
            title="Hypertuning Logs"
        />
    );
};

export default HypertuningLogs;
