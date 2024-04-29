import './App.css';
import {BrowserRouter, Routes, Route} from 'react-router-dom'
import Home from './pages/Home'
import NoPage from './pages/NoPage';
import Progress from './pages/Progress';
import Train from './pages/Train';
import DataPrep from './pages/DataPrep';
import Saved from './pages/Saved'

const App = () => {
  return (
    <>
    <BrowserRouter>
      <Routes>
        <Route index element={<Home/>}></Route>
        <Route path='/progress' element={<Progress/>}></Route>
        <Route path='/train' element={<Train/>}></Route>
        <Route path='/data-prep' element={<DataPrep/>}></Route>
        <Route path='/saved' element={<Saved/>}></Route>
        <Route path='*' element={<NoPage/>}></Route>
      </Routes>
    </BrowserRouter>
    </>
  );
}

export default App;
