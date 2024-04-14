import './App.css';
import  Chat  from './components/Chat';

import { BrowserRouter as Router, Routes, Route} from 'react-router-dom';

const App = () => {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Chat />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
