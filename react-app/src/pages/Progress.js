import '../App.css';
import { NavLink } from 'react-router-dom';
import home from '../images/home.png';
import saved from '../images/saved.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import NavBar from '../components/NavBar';
import { useState } from 'react';

const Progress = () => {  
  const [currentMenu, setCurrentMenu] = useState("Logs");  // State to track the selected menu
  const [logs, setLogs] = useState(["Here are your logs..."]);
  const [loading, setLoading] = useState(false);

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
    switch (currentMenu) {
      case "Logs":
        return (
          <div className='nav-content'>
            {logs.map((log, index) => (
              <p key={index}>{log}</p> // Each log line is rendered as a paragraph
            ))}
          </div>
        );
      case "Results":
        return <p className='nav-content'>Here are your results...</p>;
      case "Details":
        return <p className='nav-content'>More details will be shown here...</p>;
      case "Parameters":
        return <p className='nav-content'>Adjust your parameters here...</p>;
      default:
        return <p className='nav-content'>Content goes here...</p>;
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
