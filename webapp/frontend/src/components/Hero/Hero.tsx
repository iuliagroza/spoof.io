import React from 'react';
import './Hero.scss';

const Hero: React.FC = () => {
    return (
        <div className="Hero">
            <div className="Hero-content">
                <h1 className="Hero-title">Secure Your LUNA Transactions Against Spoofing</h1>
                <p className="Hero-description">
                    Protect yourself from market manipulations with real-time monitoring of spoofing & layering attempts on LUNA.
                </p>
            </div>
        </div>
    );
};

export default Hero;
