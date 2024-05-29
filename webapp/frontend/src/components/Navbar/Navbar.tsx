import React from 'react';
import { Link } from 'react-router-dom';
import './Navbar.scss';

const Navbar: React.FC<{ setNavbarHeight: (height: number) => void }> = ({ setNavbarHeight }) => {
    const navbarRef = React.useRef<HTMLDivElement>(null);

    React.useEffect(() => {
        if (navbarRef.current) {
            setNavbarHeight(navbarRef.current.clientHeight);
        }
    }, [setNavbarHeight]);

    return (
        <nav className="navbar" ref={navbarRef}>
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
