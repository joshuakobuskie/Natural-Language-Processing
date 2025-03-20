import React, { useState, useRef, useEffect } from 'react';

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [dots, setDots] = useState('');
  const messagesEndRef = useRef(null);

  // Dot animation effect
  useEffect(() => {
    let interval;
    if (isLoading) {
      interval = setInterval(() => {
        setDots(prev => {
          if (prev.length >= 3) return '';
          return prev + '.';
        });
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isLoading]);

  // Scrolls to the bottom of the messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {

    // Prevents the submission of empty messages
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message and show loading sequence
    setMessages(prev => [...prev, { text: input, isUser: true }]);
    setInput('');
    setIsLoading(true);

    try {
      // Calls backend API
      const response = await fetch('http://localhost:5001/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: input }),
      });

      // Display if the API call fails
      if (!response.ok) throw new Error('RAG Chat was unable to connect to the server. Please try again later');
      
      // Display the AI response
      const data = await response.json();
      setMessages(prev => [...prev, { text: data.response, isUser: false }]);

    } catch (error) {
      setMessages(prev => [...prev, { text: 'Error: Could not generate a response', isUser: false }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="title-container">RAG Chat System</div>
      <div className="messages-container">
        {messages.map((msg, index) => (
          <div 
            key={index}
            className={`message ${msg.isUser ? 'user' : 'ai'}`}
          >
            {msg.text}
          </div>
        ))}
        {isLoading && (
          <div className="message ai loading">RAG Chat is thinking{dots}</div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          Send
        </button>
      </form>
    </div>
  );
};

export default ChatInterface;