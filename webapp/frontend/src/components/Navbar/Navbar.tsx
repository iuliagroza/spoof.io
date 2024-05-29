import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.scss';

const Navbar: React.FC = () => {
    return (
        <nav className="navbar">
            <h1 className="navbar-logo">spoof.io</h1>
            <div className="navbar-links">
                <Link to="/" className="navbar-link">Detector</Link>
                <Link to="/training-logs" className="navbar-link">Training Logs</Link>
                <Link to="/hypertuning-logs" className="navbar-link">Hypertuning Logs</Link>
            </div>
        </nav>
    );
};

export default Navbar;
