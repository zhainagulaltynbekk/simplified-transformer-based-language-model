import '../App.css';
import { NavLink } from 'react-router-dom';
import chat from '../images/chat.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import dataPrep from '../images/data-prep.png';
import NavBar from '../components/NavBar';
import { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import 'chart.js/auto';

const Progress = () => {  
  const [currentMenu, setCurrentMenu] = useState("Logs");
  const [logs, setLogs] = useState([{ message: "Logs will appear here ...", time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }) }]);
  const [results, setResults] = useState([{ message: "Results of the training will appear here ...", time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }) }]);
  const [parameters, setParameters] = useState([{ message: "Parameters used for this training will be listed here ...", time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }) }]);
  const [samples, setSamples] = useState([{ message: "Sample trained text generated will be here ...", time: new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true }) }]);
  const [loading, setLoading] = useState(false);
  const [currentDate, setCurrentDate] = useState('');
  const [imageSrcs, setImageSrcs] = useState([]);
  const [currentImageIndex, setCurrentImageIndex] = useState(0);
  const [chartData, setChartData] = useState({
    labels: [],
    datasets: [
      {
        label: 'Training Loss',
        data: [],
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true,
      },
      {
        label: 'Validation Loss',
        data: [],
        borderColor: 'rgba(153, 102, 255, 1)',
        backgroundColor: 'rgba(153, 102, 255, 0.2)',
        fill: true,
      },
      {
        label: 'Model Loss',
        data: [],
        borderColor: 'rgba(255, 99, 132, 1)',
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        fill: true,
      }
    ]
  });

  useEffect(() => {
    const date = new Date();
    setCurrentDate(date.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    }));
  }, []);

  const handleMenuChange = (menu) => {
    setCurrentMenu(menu);
  };

  const fetchLogs = async () => {
    setLoading(true);
    const now = new Date().toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
      hour12: true
    });
    setLogs([{ message: "Training process started.", time: now }]);
    setResults([{ message: "Training process started. Results of the training process will be here.", time: now }]);
    setParameters([{ message: "Training process started. Parameters of the training will appear here.", time: now }]);
    setSamples([{ message: "The below you can see the text sample text generated after the current training process.", time: now }]);
    setImageSrcs([]);
    setCurrentImageIndex(0);

    const eventSource = new EventSource('http://localhost:5000/model-train');

    let sampleBuffer = [];
    let isSampleBlock = false;

    eventSource.onmessage = function(event) {
      const now = new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: true
      });

      if (event.data === "done") {
        setLogs(prevLogs => [...prevLogs, { message: "Training completed", time: now }]);
        setLoading(false);
        eventSource.close();

        const fetchImages = async () => {
          for (let i = 0; i <= 9; i += 1) {
            const response = await fetch(`http://localhost:5000/images/confusion_matrix_${i}.png?${new Date().getTime()}`);
            if (response.ok) {
              const src = `http://localhost:5000/images/confusion_matrix_${i}.png?${new Date().getTime()}`;
              setImageSrcs(prevImageSrcs => [...prevImageSrcs, {src, time: now}]);
            }
          }
        };

        fetchImages();
      } else {
        const line = event.data;
        const cleanLine = line.replace(/^LOG: |^RESULT: |^PARAM: /, '');

        if (line.startsWith("SAMPLE_BLOCK_START")) {
          isSampleBlock = true;
          sampleBuffer = [];
        } else if (line.startsWith("SAMPLE_BLOCK_END")) {
          isSampleBlock = false;
          setSamples(prevSamples => [...prevSamples, { message: sampleBuffer.join('\n'), time: now }]);
        } else if (isSampleBlock) {
          sampleBuffer.push(line);
        } else {
          const logEntry = { message: cleanLine, time: now };

          if (line.startsWith("LOG:")) {
            setLogs(prevLogs => [...prevLogs, logEntry]);
          } else if (line.startsWith("RESULT:")) {
            if (cleanLine.includes('train loss') || cleanLine.includes('validation loss') || cleanLine.includes('model loss')) {
              const [step, trainLoss, valLoss, modelLoss] = cleanLine.match(/step (\d+), train loss ([\d.]+), validation loss ([\d.]+), model loss ([\d.]+)/).slice(1, 5);
              setChartData(prevChartData => ({
                ...prevChartData,
                labels: [...prevChartData.labels, `Step ${step}`],
                datasets: prevChartData.datasets.map(dataset => {
                  if (dataset.label === 'Training Loss') return { ...dataset, data: [...dataset.data, trainLoss] };
                  if (dataset.label === 'Validation Loss') return { ...dataset, data: [...dataset.data, valLoss] };
                  if (dataset.label === 'Model Loss') return { ...dataset, data: [...dataset.data, modelLoss] };
                  return dataset;
                })
              }));
            }
            setResults(prevResults => [...prevResults, logEntry]);
            setLogs(prevLogs => [...prevLogs, logEntry]);
          } else if (line.startsWith("PARAM:")) {
            setParameters(prevParams => [...prevParams, logEntry]);
            setLogs(prevLogs => [...prevLogs, logEntry]);
          } else {
            setLogs(prevLogs => [...prevLogs, logEntry]);
          }
        }
      }
    };

    eventSource.onerror = function(err) {
      console.error('EventSource failed:', err);
      setLoading(false);
      eventSource.close();
    };

    eventSource.onopen = function() {
      console.log('Connection to server opened.');
    };

    return () => {
      eventSource.close();
    };
  };

  const nextImage = () => {
    setCurrentImageIndex(prevIndex => (prevIndex + 1) % imageSrcs.length);
  };

  const prevImage = () => {
    setCurrentImageIndex(prevIndex => (prevIndex - 1 + imageSrcs.length) % imageSrcs.length);
  };

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
            <div className="chart-container">
              <Line data={chartData} />
            </div>
            {logs.map((log, index) => (
              <div key={index} className="progress-log">
                <span>{log.message}</span>
                <span className="upload-time-prg">{log.time}</span>
              </div>
            ))}
            {imageSrcs.length > 0 && (
              <div className="image-container">
                <div className="image-nav-buttons">
                  <button onClick={prevImage} className="image-nav-button">Previous</button>
                  <button onClick={nextImage} className="image-nav-button">Next</button>
                </div>
                <div className='progress-log'>
                  <img src={imageSrcs[currentImageIndex].src} alt={`Confusion Matrix ${currentImageIndex}`} className="confusion-matrix" />
                  <span className="upload-time-prg">{imageSrcs[currentImageIndex].time}</span>
                </div>
              </div>
            )}
          </div>
        );
      case "Results":
        return (
          <div className='nav-content'>
            {header}
            <div className="chart-container">
              <Line data={chartData} />
            </div>
            {results.map((result, index) => (
              <div key={index} className="progress-log">
                <span>{result.message}</span>
                <span className="upload-time-prg">{result.time}</span>
              </div>
            ))}
            {imageSrcs.length > 0 && (
              <div className="image-container">
                <div className="image-nav-buttons">
                  <button onClick={prevImage} className="image-nav-button">Previous</button>
                  <button onClick={nextImage} className="image-nav-button">Next</button>
                </div>
                <div className='progress-log'>
                  <img src={imageSrcs[currentImageIndex].src} alt={`Confusion Matrix ${currentImageIndex}`} className="confusion-matrix" />
                  <span className="upload-time-prg">{imageSrcs[currentImageIndex].time}</span>
                </div>
              </div>
            )}
          </div>
        );
      case "Parameters":
        return (
          <div className='nav-content'>
            {header}
            {parameters.map((param, index) => (
              <div key={index} className="progress-log">
                <span>{param.message}</span>
                <span className="upload-time-prg">{param.time}</span>
              </div>
            ))}
          </div>
        );
      case "Sample":
        return (
          <div className='nav-content'>
            {header}
            {samples.map((sample, index) => (
              <div key={index} className="progress-log">
                <span>{sample.message}</span>
                <span className="upload-time-prg">{sample.time}</span>
              </div>
            ))}
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
        <NavBar onMenuChange={handleMenuChange} />
        {renderContent()}
      </div>
    </div>
  );
}

export default Progress;
