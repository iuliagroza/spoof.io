import React from 'react';
import { NavLink } from 'react-router-dom';
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
                <NavLink to="/" className={({ isActive }) => isActive ? "navbar-link active-link" : "navbar-link"}>Detector</NavLink>
                <NavLink to="/training-logs" className={({ isActive }) => isActive ? "navbar-link active-link" : "navbar-link"}>Training Logs</NavLink>
                <NavLink to="/hypertuning-logs" className={({ isActive }) => isActive ? "navbar-link active-link" : "navbar-link"}>Hypertuning Logs</NavLink>
            </div>
        </nav>
    );
};

export default Navbar;
