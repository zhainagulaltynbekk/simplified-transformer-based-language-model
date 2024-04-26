import { useState } from 'react';
import React from 'react';
import './NavBar.css';

const NavBar = ({ onMenuChange, content }) => {
  const [menu, setMenu] = useState("Logs");

  const handleClick = (menuName) => {
    setMenu(menuName);
    onMenuChange(menuName);  // Trigger menu change in parent component
  }

  return (
    <div className='navbar'>
      <ul className='navbar-menu'>
        <li onClick={() => handleClick("Logs")} className={menu === "Logs" ? "active" : ""}>Logs</li>
        <li onClick={() => handleClick("Results")} className={menu === "Results" ? "active" : ""}>Results</li>
        <li onClick={() => handleClick("Details")} className={menu === "Details" ? "active" : ""}>Details</li>
        <li onClick={() => handleClick("Parameters")} className={menu === "Parameters" ? "active" : ""}>Parameters</li>
      </ul>
    </div>
  );
}

export default NavBar;