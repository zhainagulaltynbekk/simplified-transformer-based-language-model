import '../App.css';
import { NavLink } from 'react-router-dom';
// import home from '../images/home.png';
import chat from '../images/chat.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import dataPrep from '../images/data-prep.png';
import axios from 'axios';
import { useState, useEffect } from 'react';

const Train = () => {  
  const [formData, setFormData] = useState({
    batch_size: '32',
    block_size: '128',
    max_iters: '10',
    eval_interval: '100',
    learning_rate: '3e-4',
    device: 'cpu',
    eval_iters: '1',
    n_embd: '384',
    n_head: '8',
    n_layer: '8',
    dropout: '0.2',
    file: null
  });
  const [submittedData, setSubmittedData] = useState(null);  // State to store the submitted data
  const [currentDate, setCurrentDate] = useState('');

  useEffect(() => {
    const date = new Date();
    setCurrentDate(date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
    }));
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e) => {
    setFormData(prev => ({ ...prev, file: e.target.files[0] }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();
    // Append all text-based inputs to the FormData object from state
    Object.entries(formData).forEach(([key, value]) => {
        if (key !== 'file') {
            data.append(key, value);
        } else if (value) { // Only append file if it's not null
            data.append('file', value, value.name); // Include the file name
        }
    });

    try {
        const response = await axios.post('http://localhost:5000/submit-form', data, {
            headers: {
                'Content-Type': 'multipart/form-data'
            }
        });
        const responseData = {
            ...formData,
            fileName: formData.file ? formData.file.name : null
        };
        setSubmittedData(responseData);
        alert('Form submitted successfully!');
    } catch (error) {
        alert('Failed to submit form!');
        console.error('Error:', error);
    }
  };

  return (
    <div className="App">
      <div className='sideBar'>
        <div className='upperSide'>
          {/* <NavLink to="/" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
              <img src={home} alt='home' className='listItemsImg' />Home
          </NavLink> */}
          <NavLink to="/" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'} end>
              <img src={chat} alt='chat' className='listItemsImg' />Chat
          </NavLink>
          <NavLink to="/data-prep" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
              <img src={dataPrep} alt='data preparation' className='listItemsImg' />Data Preparation
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
        <h2 className='pageName'>Train</h2>
        <form className='param-form' onSubmit={handleSubmit}>
          <div className='form-row'>
            <div className='form-item'>
              <label className='label'>Batch Size</label>
              <input className='form-input' type="number" name="batch_size" placeholder="Batch Size" value={formData.batch_size} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Block Size</label>
              <input className='form-input' type="number" name="block_size" placeholder="Block Size" value={formData.block_size} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Max Iterations</label>
              <input className='form-input' type="number" name="max_iters" placeholder="Max Iterations" value={formData.max_iters} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Evaluation Interval</label>
              <input className='form-input' type="number" name="eval_interval" placeholder="Evaluation Interval" value={formData.eval_interval} onChange={handleChange} />
            </div>
          </div>
          <div className='form-row'>
            <div className='form-item'>
              <label className='label'>Learning Rate</label>
              <input className='form-input' type="number" name="learning_rate" placeholder="Learning Rate" value={formData.learning_rate} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Device</label>
              <input className='form-input' type="text" name="device" placeholder="Device" value={formData.device} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Evaluation Iterations</label>
              <input className='form-input' type="number" name="eval_iters" placeholder="Evaluation Iterations" value={formData.eval_iters} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Number of Embeddings</label>
              <input className='form-input' type="number" name="n_embd" placeholder="Number of Embeddings" value={formData.n_embd} onChange={handleChange} />
            </div>
          </div>
          <div className='form-row'>
            <div className='form-item'>
              <label className='label'>Number of Heads</label>
              <input className='form-input' type="number" name="n_head" placeholder="Number of Heads" value={formData.n_head} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Number of Layers</label>
              <input className='form-input' type="number" name="n_layer" placeholder="Number of Layers" value={formData.n_layer} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Dropout Rate</label>
              <input className='form-input' type="number" name="dropout" placeholder="Dropout Rate" value={formData.dropout} onChange={handleChange} />
            </div>
            <div className='form-item'>
              <label className='label'>Model File</label>
              <input className='form-file' type="file" onChange={handleFileChange} />
            </div>
          </div>
          <button type="submit" className='form-btn'>Submit</button>
        </form>
        <div className='train-content'>
          <div className="train-date">
            <div className="border-edge"></div>
            <span className="data-date">{currentDate}</span>
            <div className="border-edge"></div>
          </div>
          {submittedData && (
            <div>
              <h3 className='data-title-first'>Hyperparameters</h3>
              {Object.entries(submittedData).map(([key, value]) => (
                key !== 'file' ? <div className='data-info' key={key}>{`${key}: ${value}`}</div> : null
              ))}
              <div className='data-info'>
                {submittedData.fileName ? `File chosen: ${submittedData.fileName}` : "Model is not given, default model in model/ will be used or the new one will be created."}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Train;
