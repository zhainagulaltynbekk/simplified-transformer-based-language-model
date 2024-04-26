import '../App.css';
import { NavLink } from 'react-router-dom';
import home from '../images/home.png';
import saved from '../images/saved.png';
import rocket from '../images/rocket.png';
import progress from '../images/progress.png';
import { useState } from 'react';


const Train = () => {  
  const [formData, setFormData] = useState({
    batch_size: '',
    block_size: '',
    max_iters: '',
    eval_interval: '',
    learning_rate: '',
    device: '',
    eval_iters: '',
    n_embd: '',
    n_head: '',
    n_layer: '',
    dropout: '',
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
          <label>Batch Size</label>
          <input className='form-item' type="number" name="batch_size" placeholder="Batch Size" value={formData.batch_size} onChange={handleChange} />

          <label>Block Size</label>
          <input className='form-item' type="number" name="block_size" placeholder="Block Size" value={formData.block_size} onChange={handleChange} />

          <label>Max Iterations</label>
          <input className='form-item' type="number" name="max_iters" placeholder="Max Iterations" value={formData.max_iters} onChange={handleChange} />

          <label>Evaluation Interval</label>
          <input className='form-item' type="number" name="eval_interval" placeholder="Evaluation Interval" value={formData.eval_interval} onChange={handleChange} />

          <label>Learning Rate</label>
          <input className='form-item' type="number" name="learning_rate" placeholder="Learning Rate" value={formData.learning_rate} onChange={handleChange} />

          <label>Device</label>
          <input className='form-item' type="text" name="device" placeholder="Device" value={formData.device} onChange={handleChange} />

          <label>Evaluation Iterations</label>
          <input className='form-item' type="number" name="eval_iters" placeholder="Evaluation Iterations" value={formData.eval_iters} onChange={handleChange} />

          <label>Number of Embeddings</label>
          <input className='form-item' type="number" name="n_embd" placeholder="Number of Embeddings" value={formData.n_embd} onChange={handleChange} />

          <label>Number of Heads</label>
          <input className='form-item' type="number" name="n_head" placeholder="Number of Heads" value={formData.n_head} onChange={handleChange} />

          <label>Number of Layers</label>
          <input className='form-item' type="number" name="n_layer" placeholder="Number of Layers" value={formData.n_layer} onChange={handleChange} />

          <label>Dropout Rate</label>
          <input className='form-item' type="number" name="dropout" placeholder="Dropout Rate" value={formData.dropout} onChange={handleChange} />

          <label>Model File</label>
          <input type="file" onChange={handleFileChange} />

          <button type="submit">Submit</button>
        </form>
        </div>
      </div>
  );
}

export default Train;
