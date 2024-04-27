import '../App.css';
import { NavLink } from 'react-router-dom';
import home from '../images/home.png';
import saved from '../images/saved.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import { useState } from 'react';


const Train = () => {  
  const [formData, setFormData] = useState({
    batch_size: '32', // Default value
    block_size: '128', // Default value
    max_iters: '10', // Default value
    eval_interval: '100', // Default value
    learning_rate: '3e-4', // Default value
    device: 'cpu', // Default value
    eval_iters: '1', // Default value
    n_embd: '384', // Default value
    n_head: '8', // Default value
    n_layer: '8', // Default value
    dropout: '0.2', // Default value
    file: null
  });  
  
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };
  
  const handleFileChange = (e) => {
    setFormData(prev => ({ ...prev, file: e.target.files[0] }));
  };
  
  const handleSubmit = (e) => {
    e.preventDefault();
    // Here you would typically handle the submission to the backend
    console.log(formData);
    alert('Form submitted. Check the console for data.');
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
        </div>
      </div>
  );
}

export default Train;
