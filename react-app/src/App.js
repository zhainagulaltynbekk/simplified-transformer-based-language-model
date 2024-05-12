import './App.css';
import {BrowserRouter, Routes, Route} from 'react-router-dom'
import Chat from './pages/Chat'
import NoPage from './pages/NoPage';
import Progress from './pages/Progress';
import Train from './pages/Train';
import DataPrep from './pages/DataPrep';
import Home from './pages/Home';

const App = () => {
  return (
    <>
    <BrowserRouter>
      <Routes>
        <Route index element={<Home/>}></Route>
        <Route path='/progress' element={<Progress/>}></Route>
        <Route path='/train' element={<Train/>}></Route>
        <Route path='/data-prep' element={<DataPrep/>}></Route>
        <Route path='/chat' element={<Chat/>}></Route>
        <Route path='*' element={<NoPage/>}></Route>
      </Routes>
    </BrowserRouter>
    </>
  );
}

export default App;
