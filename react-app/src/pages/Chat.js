import { useState, useRef, useEffect } from "react";
import { NavLink } from 'react-router-dom';
import '../App.css';
// import home from '../images/home.png';
import chat from '../images/chat.png';
import rocket from '../images/rocket.png';
import sendBtn from '../images/send.png';
import userIcon from '../images/user.png';
import botIcon from '../images/round.png';
import spinner from '../images/load.gif';
import progress from '../images/progress.png';
import msgIcon from '../images/msg.png';
import dataPrep from '../images/data-prep.png'


const Chat = () => {
  const msgEnd = useRef(null);
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([
    {
      text: "Hi, I am simplified language model developed for web based user interaction, please, ask your question!",
      time: (`${new Date().getHours().toString().padStart(2, '0')}:${new Date().getMinutes().toString().padStart(2, '0')}`),
      sender: 'bot',
    }
  ]);

  useEffect(()=> {
    msgEnd.current.scrollIntoView();
  },[messages])

  const sendMessage = async (rawText) => {
    const date = new Date();
    const str_time = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    
    const newUserMessage = {
        text: rawText,
        time: str_time,
        sender: 'user',
    };

    // Add user message to state
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);

    // Initialize a bot message for the spinner
    let newBotMessage = {
      text: <img src={spinner} alt="Loading..." className="img-fluid"/>,
      time: str_time,
      sender: 'bot',
    };

    // Add loading spinner as bot message
    setMessages((prevMessages) => [...prevMessages, newBotMessage]);

    const response = await fetch('http://localhost:5000/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ msg: rawText }),
    });

    const reader = response.body.getReader();
    let receivedText = '';

    const processText = async ({ done, value }) => {
      if (done) {
        console.log('Stream complete');
        return;
      }
      const text = new TextDecoder("utf-8").decode(value);
      receivedText += text;

      // Update the bot message with new text or remove spinner
      setMessages((prevMessages) => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'bot') {
          return [...prevMessages.slice(0, -1), { ...lastMessage, text: receivedText }];
        } else {
          return [...prevMessages, {
            text: receivedText,
            time: str_time,
            sender: 'bot',
          }];
        }
      });

      return reader.read().then(processText);
    };

    reader.read().then(processText).catch(error => {
      console.error("Failed to read stream", error);
    });
  };

  const onClickHandler = (event) => {
    event.preventDefault();
      if (message.trim()) {
        sendMessage(message);
        setMessage("");
      }
  };

  const onChangeInputHandler = (e) => {
    setMessage(e.target.value);
  };

  const handleQuery = async (e) => {
    const date = new Date();
    const str_time = `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}`;
    
    const newUserMessage = {
        text: e.target.value,
        time: str_time,
        sender: 'user',
    };

    // Add user message to state
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);

    // Initialize a bot message for the spinner
    let newBotMessage = {
      text: <img src={spinner} alt="Loading..." className="img-fluid"/>,
      time: str_time,
      sender: 'bot',
    };

    // Add loading spinner as bot message
    setMessages((prevMessages) => [...prevMessages, newBotMessage]);

    const response = await fetch('http://localhost:5000/stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ msg: e.target.value }),
    });

    const reader = response.body.getReader();
    let receivedText = '';

    const processText = async ({ done, value }) => {
      if (done) {
        console.log('Stream complete');
        return;
      }
      const text = new TextDecoder("utf-8").decode(value);
      receivedText += text;

      // Update the bot message with new text or remove spinner
      setMessages((prevMessages) => {
        const lastMessage = prevMessages[prevMessages.length - 1];
        if (lastMessage && lastMessage.sender === 'bot') {
          return [...prevMessages.slice(0, -1), { ...lastMessage, text: receivedText }];
        } else {
          return [...prevMessages, {
            text: receivedText,
            time: str_time,
            sender: 'bot',
          }];
        }
      });

      return reader.read().then(processText);
    };

    reader.read().then(processText).catch(error => {
      console.error("Failed to read stream", error);
    });
  }
  
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
                <img src={dataPrep} alt='data preperation' className='listItemsImg' />Data Preperation
            </NavLink>
            <NavLink to="/train" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={rocket} alt='train' className='listItemsImg' />Train
            </NavLink>
            <NavLink to="/progress" className={({ isActive }) => isActive ? 'listItems activeLink' : 'listItems'}>
                <img src={progress} alt='upgrade' className='listItemsImg' />Progress
            </NavLink>
          </div>
        </div>
        <div className='mainHome'>
        <h2 className="pageNameHome">Chat<button className='midBtn' onClick={()=>{window.location.reload()}}>+</button></h2>        
          <div className="chats">
            {messages.map((msg, index) => (
              <div key={index} className={`chat${msg.sender === 'user' ? '' : '-bot'}`}>
                {msg.sender === 'bot' && (
                  <img src={botIcon} className="chatImg" alt="Bot"/>
                )}
                {msg.sender === 'user' && (
                  <img src={userIcon} className="chatImg" alt="User"/>
                )}
                <p className='txt'>
                  {typeof msg.text === 'string' ? msg.text : msg.text}
                  <span className='msg_time'>{msg.time}</span>
                </p>
              </div>
            ))}
            <div ref={msgEnd}/>
          </div>
          <div className='chatFooter'>
            <form className='inp' onSubmit={onClickHandler}>
              <input type="text" placeholder='Type your message...' value={message} autoComplete="off" required onChange={onChangeInputHandler}/><button type='submit' className='send'><img src={sendBtn} alt='Send message'/></button>
            </form>
            <p>This ChatBot may produce incorrect, innacurate information about people, places, or facts.</p>
          </div>
        </div>
        <div className="sideBarR">
            <div className='upperSideBottom'>
              <button className='query' onClick={handleQuery} value={"Hello! Could you help?"}><img src={msgIcon} alt='query'/>Hello! Could you help?</button>
              <button className='query' onClick={handleQuery} value={"Thank you for the help!"}><img src={msgIcon} alt='query'/>Thank you for the help!</button>
            </div>
        </div>
      </div>
  );
}

export default Chat;