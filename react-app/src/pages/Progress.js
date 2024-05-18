import '../App.css';
import { NavLink } from 'react-router-dom';
// import home from '../images/home.png';
import chat from '../images/chat.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import dataPrep from '../images/data-prep.png'
import NavBar from '../components/NavBar';
import { useState, useEffect } from 'react';

const Progress = () => {  
  const [currentMenu, setCurrentMenu] = useState("Logs");  // State to track the selected menu
  const [logs, setLogs] = useState(["Logs will appear here ..."]);
  const [results, setResults] = useState(["Results of the training will appear here ..."]);
  const [parameters, setParameters] = useState(["Parameters used for this training will be listed here ..."]);
  const [samples, setSample] = useState(["Sample trained text generated will be here ..."]);
  const [loading, setLoading] = useState(false);
  const [currentDate, setCurrentDate] = useState('');
  const [uploadTime, setUploadTime] = useState('');

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
    setLogs(["Training process started."]); // Log that the training process has started
    setResults(["Training process started. Results of the training process will be here."]);
    setParameters(["Training process started. Parameters of the training will appear here."]);
    setSample(["The below you can see the text sample text generated after the current training process."]);

    const now = new Date();
    setUploadTime(now.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    }));
    
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
      const decoder = new TextDecoder('utf-8');
  
      let done, value;
      let isSampleBlock = false;
      let sampleBuffer = [];
      while (!done) {
        ({ done, value } = await reader.read());
        if (value) {
          const logContent = decoder.decode(value, { stream: true }).replace(/\r\n?/g, "\n").split("\n");
          logContent.forEach(line => {
            if (line.startsWith("SAMPLE_BLOCK_START")) {
              isSampleBlock = true;
              sampleBuffer = [];
            } else if (line.startsWith("SAMPLE_BLOCK_END")) {
              isSampleBlock = false;
              setSample(prevSample => [...prevSample, sampleBuffer.join('\n')]);  // Join buffer and update sample
            } else if (isSampleBlock) {
              sampleBuffer.push(line);
            } else {
              if (line.startsWith("LOG:")) {
                setLogs(prevLogs => [...prevLogs, line.substring(5)]);
              } else if (line.startsWith("RESULT:")) {
                setResults(prevResults => [...prevResults, line.substring(8)]);
              } else if (line.startsWith("PARAM:")) {
                setParameters(prevParams => [...prevParams, line.substring(7)]);
              }
            }
          });
        }
      }
  
      setLogs(prevLogs => [...prevLogs, "Training completed"]); // Log that the training is completed
    } catch (error) {
      console.error('Failed to fetch logs:', error.message);
      setLogs(prevLogs => [...prevLogs, "Failed to fetch logs."]);
    } finally {
      setLoading(false);
    }
  };
  
  // Render the corresponding text based on the selected menu
  const renderContent = () => {
    const header = (
      <div className="nav-header">
        <div className="border-edge"></div>
        <span className="nav-date">{currentDate}</span>
        <div className="border-edge"></div>
      </div>
    );

    switch (currentMenu) {
      case "Logs":
        return (
          <div className='nav-content'>
            {header}
            {logs.map((log, index) => <p className="progress-log" key={index}>{log}<span className="upload-time-prg">{uploadTime}</span></p>)}
          </div>
        );
      case "Results":
        return (
          <div className='nav-content'>
            {header}
            {results.map((result, index) => <p className="progress-log" key={index}>{result}<span className="upload-time-prg">{uploadTime}</span></p>)}
          </div>
        );
      case "Parameters":
        return (
          <div className='nav-content'>
            {header}
            {parameters.map((param, index) => <p className="progress-log" key={index}>{param}<span className="upload-time-prg">{uploadTime}</span></p>)}
          </div>
        );
      case "Sample":
        return (
          <div className='nav-content'>
            {header}
            <p className='progress-log'>Training process started.<span className="upload-time-prg">{uploadTime}</span></p>
            {samples.map((sample, index) => <p className="progress-log" key={index}>{sample}</p>)}
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
  };


  return (
      <div className="App">
        <div className='sideBar'>
          <div className='upperSide'>
            {/* <NavLink to="/" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={home} alt='saved' className='listItemsImg' />Home
            </NavLink> */}
            <NavLink to="/" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'} end>
                <img src={chat} alt='chat' className='listItemsImg' />Chat
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
