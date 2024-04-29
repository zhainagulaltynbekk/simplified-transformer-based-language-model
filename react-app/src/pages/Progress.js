import '../App.css';
import { NavLink } from 'react-router-dom';
import home from '../images/home.png';
import saved from '../images/saved.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import dataPrep from '../images/data-prep.png'
import NavBar from '../components/NavBar';
import { useState, useEffect } from 'react';

const Progress = () => {  
  const [currentMenu, setCurrentMenu] = useState("Logs");  // State to track the selected menu
  const [logs, setLogs] = useState(["Here are your logs..."]);
  const [loading, setLoading] = useState(false);
  const [currentDate, setCurrentDate] = useState('');

    useEffect(() => {
        // Get the current date in the desired format
        const date = new Date();
        setCurrentDate(date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        }));
    }, []);

  const handleMenuChange = (menu) => {
    setCurrentMenu(menu);  // Update the current menu state
  }

  const fetchLogs = async () => {
    setLoading(true);
    setLogs(["Training process started"]); // Log that the training process has started
    
    try {
      const response = await fetch('http://localhost:5000/model-train', {
        method: 'POST',
        headers: {
          'Accept': 'text/plain',
          'Content-Type': 'application/json'
        }
      });
  
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
  
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
  
      let done, value;
      while (!done) {
        ({ done, value } = await reader.read());
        if (value) {
          const logContent = decoder.decode(value, { stream: true });
          setLogs(prevLogs => [...prevLogs, logContent]);
        }
      }
  
      setLogs(prevLogs => [...prevLogs, "Training completed"]); // Log that the training is completed
      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch logs:', error.message);
      setLogs(prevLogs => [...prevLogs, "Failed to fetch logs."]);
      setLoading(false);
    }
  };
  
  // Render the corresponding text based on the selected menu
  const renderContent = () => {
    const header = (
      <div className="nav-header">
        <div className="border-edge"></div> {/* Left border */}
        <span className="nav-date">{currentDate}</span>
        <div className="border-edge"></div> {/* Right border */}
      </div>
    );

    switch (currentMenu) {
      case "Logs":
        return (
          <div className='nav-content'>
            {header}
            {logs.map((log, index) => (
              <p key={index}>{log}</p>
            ))}
          </div>
        );
      case "Results":
        return (
          <div className='nav-content'>
            {header}
            <p>Here are your results...</p>
          </div>
        );
      case "Details":
        return (
          <div className='nav-content'>
            {header}
            <p>More details will be shown here...</p>
          </div>
        );
      case "Parameters":
        return (
          <div className='nav-content'>
            {header}
            <p>Adjust your parameters here...</p>
          </div>
        );
      default:
        return (
          <div className='nav-content'>
            {header}
            <p>Content goes here...</p>
          </div>
        );
    }
  }

  return (
      <div className="App">
        <div className='sideBar'>
          <div className='upperSide'>
            <NavLink to="/" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'} end>
                <img src={home} alt='chat' className='listItemsImg' />Chat
            </NavLink>
            <NavLink to="/saved" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={saved} alt='saved' className='listItemsImg' />Saved
            </NavLink>
            <NavLink to="/data-prep" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={dataPrep} alt='data preperation' className='listItemsImg' />Data Preperation
            </NavLink>
            <NavLink to="/train" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={rocket} alt='train' className='listItemsImg' />Train
            </NavLink>
            <NavLink to="/progress" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={progress} alt='progress' className='listItemsImg' />Progress
            </NavLink>
          </div>
        </div>
        <div className="main">
          <h2 className='pageName'>Progress</h2>
          <button className='trainBtn' onClick={fetchLogs}>{loading ? "Training..." : "Train"}</button>
          <NavBar onMenuChange={handleMenuChange}/>
          {renderContent()}
        </div>
      </div>
  );
}

export default Progress;
