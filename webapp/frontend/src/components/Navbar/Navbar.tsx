import React from 'react';
import './Navbar.scss';

const Navbar = () => {
    return (
        <nav className="navbar">
            <h1 className="navbar-logo">spoof.io</h1>
            <div className="navbar-links">
                <a href="/test_results.html" target="_blank" className="navbar-link">Training Logs</a>
            </div>
        </nav>
    );
};

export default Navbar;
