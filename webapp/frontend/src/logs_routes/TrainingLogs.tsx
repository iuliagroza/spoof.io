import React from 'react';

const TrainingLogs: React.FC = () => {
    return (
        <iframe
            src="/test_results.html"
            style={{ width: '100%', height: '80vh', border: 'none' }}
            title="Training Logs"
        />
    );
};

export default TrainingLogs;
