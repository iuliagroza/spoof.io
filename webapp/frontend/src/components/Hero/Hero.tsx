import React, { useRef, useEffect } from 'react';
import './Hero.scss';

const Hero: React.FC<{ setHeroHeight: (height: number) => void }> = ({ setHeroHeight }) => {
    const heroRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (heroRef.current) {
            setHeroHeight(heroRef.current.clientHeight);
        }
    }, [setHeroHeight]);

    return (
        <div className="hero" ref={heroRef}>
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
