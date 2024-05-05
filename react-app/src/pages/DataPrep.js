import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import home from '../images/home.png';
import saved from '../images/saved.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import dataPrep from '../images/data-prep.png';
import '../App.css';

const DataPrep = () => {
    const [files, setFiles] = useState([]);
    const [result, setResult] = useState(null);
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

    const handleFileChange = (event) => {
        setFiles([...event.target.files]);
    };

    const handleSubmit = async (event) => {
        event.preventDefault();
        if (files.length > 0) {
            const formData = new FormData();
            for (const file of files) {
                formData.append('files', file);
            }
            try {
                const response = await fetch('http://localhost:5000/upload-files', {
                    method: 'POST',
                    body: formData,
                });
                const data = await response.json();
                setResult(data);
                // Set the upload time to now
                const now = new Date();
                setUploadTime(now.toLocaleTimeString('en-US', {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: true
                }));
            } catch (error) {
                alert(`Failed to send files: ${error}`);
                setResult(null); // Clear previous results if the fetch fails
            }
        } else {
            alert('Please select files first.');
        }
    };

    const columnsPerRow = 35;

    // Helper function to chunk the vocabulary array
    const chunkArray = (array, size) => {
        const result = [];
        for (let i = 0; i < array.length; i += size) {
            result.push(array.slice(i, i + size));
        }
        return result;
    };

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
                <h2 className='pageName'>Data Preparation</h2>
                <form className='param-form' onSubmit={handleSubmit}>
                    <div className='form-row'>
                        <div className='form-item'>
                            <label className='label'>Select Files </label>
                            <input className='form-file' type="file" onChange={handleFileChange} multiple />
                        </div>
                        <div className='form-item'>
                            <button className='data-btn' type="submit">Submit</button>
                        </div>
                    </div>
                </form>
                <div className="data-logs">
                    <div className="data-header">
                        <div className="border-edge"></div>
                        <span className="data-date">{currentDate}</span>
                        <div className="border-edge"></div>
                    </div>
                    {result ? (
                        <div>
                            <h3 className='data-title-first'>Uploaded Files</h3>
                            <p className='data-info'>{result.message}:<span className="upload-time">{uploadTime}</span></p>
                            <ul className='data-files'>
                                {result.uploadedFiles.map(file => (
                                    <li className='data-file' key={file}>{file}</li>
                                ))}
                            </ul>
                            <p className='data-info'>Uploaded files will be saved in {result.upload_folder}</p>
                            <h3 className='data-title'>Train Split</h3>
                            <p className='data-info'>{result.train_file_percentage}% of the uploaded files will be used as a training data<span className="upload-time">{uploadTime}</span></p>
                            <p className='data-info'>Training split will be saved in {result.output_file_train}</p>
                            <p className='data-info'>Train Text Length: {result.trainLength}</p>
                            <h3 className='data-title'>Validation Split</h3>
                            <p className='data-info'>{result.val_file_percentage}% of the uploaded files will be used as a validation data<span className="upload-time">{uploadTime}</span></p>
                            <p className='data-info'>Validation split will be saved in {result.output_file_val} data/val_test.txt</p>
                            <p className='data-info'>Validation Text Length: {result.valLength}</p> 
                            <h3 className='data-title'>Vocabulary</h3>
                            <p className='data-info'>Vocabulary file will be saved in {result.vocab_file}<span className="upload-time">{uploadTime}</span></p>
                            <p className='data-info'>Vocabulary Length: {result.vocabLength}</p>
                            <table className="data-table">
                                <tbody>
                                    {chunkArray(result.vocab, columnsPerRow).map((row, idx) => (
                                        <tr key={idx}>
                                            {row.map((char, charIdx) => (
                                                <td key={charIdx}>{char}</td>
                                            ))}
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    ) : (
                        <p>No results to display. Upload files to see results here.</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default DataPrep;