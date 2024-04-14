import { useState } from "react";
import 'bootstrap/dist/css/bootstrap.min.css';
import 'font-awesome/css/font-awesome.min.css';
import '@fortawesome/fontawesome-free/css/all.min.css';
import "./Chat.css";
import robot_logo from '../images/round.png';
import user_logo from '../images/user.png';
import spinner from '../images/484.gif'; // Ensure this is the correct path to your spinner image

const Chat = () => {
    const [message, setMessage] = useState("");
    const [messages, setMessages] = useState([]);

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

    return (
        <>
            <div className="container-fluid h-100">
                <div className="row justify-content-center h-100">
                    <div className="col-md-8 col-xl-6 chat">
                        <div className="card">
                            <div className="card-header msg_head">
                                <div className="d-flex bd-highlight">
                                    <div className="img_cont">
                                        <img src={robot_logo} className="rounded-circle user_img" alt="User"/>
                                        <span className="online_icon"></span>
                                    </div>
                                    <div className="user_info">
                                        <span>ChatBot</span>
                                        <p>Ask me anything!</p>
                                    </div>
                                </div>
                            </div>
                            <div className="card-body msg_card_body">
                                {messages.map((msg, index) => (
                                    <div key={index} className={`d-flex justify-content-${msg.sender === 'user' ? 'end' : 'start'} mb-4`}>
                                        {msg.sender === 'bot' && (
                                            <div className="img_cont_msg">
                                                <img src={robot_logo} className="rounded-circle user_img_msg" alt="Bot"/>
                                            </div>
                                        )}
                                        <div className={`msg_cotainer${msg.sender === 'user' ? '_send' : ''}`}>
                                            {typeof msg.text === 'string' ? msg.text : msg.text}
                                            <span className={`msg_time${msg.sender === 'user' ? '_send' : ''}`}>{msg.time}</span>
                                        </div>
                                        {msg.sender === 'user' && (
                                            <div className="img_cont_msg">
                                                <img src={user_logo} className="rounded-circle user_img_msg" alt="User"/>
                                            </div>
                                        )}
                                    </div>
                                ))}
                            </div>
                            <div className="card-footer">
                                <form id="messageArea" className="input-group" onSubmit={onClickHandler}>
                                    <input type="text" id="text" name="msg" placeholder="Type your message..." value={message} autoComplete="off" className="form-control type_msg" required onChange={onChangeInputHandler} />
                                    <div className="input-group-append">
                                        <button type="submit" id="send" className="input-group-text send_btn"><i className="fas fa-arrow-up"></i></button>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </>
    );
};

export default Chat;
