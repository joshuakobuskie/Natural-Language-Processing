import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import Switch from "react-switch";

const ChatInterface = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [dots, setDots] = useState('');
  const [rag, setRag] = useState(false);
  const [show, setShow] = useState(false);
  const [topK, setTopK] = useState(5);
  const [historyWindow, setHistoryWindow] = useState(10);
  const [filter, setFilter] = useState(true);
  const [similarityThreshold, setSimilarityThreshold] = useState(0.4);
  const [bm25, setBM25] = useState(false);
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
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = async (e) => {

    // Prevents the submission of empty messages
    if (e !== undefined) {
      e.preventDefault();
    }

    if (!input.trim()) return;

    // Add user message and show loading sequence
    setMessages(prev => [...prev, { text: input, isUser: true }]);
    setInput('');
    setIsLoading(true);

    try {
      // Calls backend API
      const response = await fetch('http://' + window.location.host.slice(0, -5) + ':5001/api/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: input, rag: rag, history: messages, topK: topK, historyWindow: historyWindow, filter: filter, similarityThreshold: similarityThreshold, bm25: bm25}),
      });

      // Display the AI response
      const data = await response.json();
      setMessages(prev => [...prev, { text: data.response, isUser: false }]);

    } catch (error) {
      setMessages(prev => [...prev, { text: 'RAG Chat was unable to connect to the server. Please try again later', isUser: false }]);
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
            <ReactMarkdown components={{p: ({ children }) => <>{children}</>,}}>{msg.text}</ReactMarkdown>
          </div>
        ))}
        {isLoading && (
          <div className="message ai loading">RAG Chat is thinking{dots}</div>
        )}
        <div ref={messagesEndRef} />
      </div>
      
      <form onSubmit={handleSubmit} className="input-form">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask me anything..."
          disabled={isLoading}
          rows = {2}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              // e.preventDefault();
              handleSubmit();
            }
          }}
        />
        <div className="button-container">
          <div className="switch-container">
            <div className="switch-row-clickable">
              <span onClick={() => setShow(!show)}><u>Advanced</u></span>
            </div>
            <div className="switch-row">
              <span>RAG:</span>
              <Switch onChange={() => setRag(!rag)} checked={rag} className="react-switch" />
            </div>
            {show && (
            <div className="switch-container">
              <label className="form-label">Top K:</label>
              <input className="form-input" type="number" min="0" max="10" step="1" value={topK.toString()} onChange={(e) => { if (Number(e.target.value) >= 0 && Number(e.target.value) <= 10) {setTopK(Number(e.target.value))}}}></input>
              <label className="form-label">History Window:</label>
              <input className="form-input" type="number" min="0" max="100" step="1" value={historyWindow.toString()} onChange={(e) => { if (Number(e.target.value) >= 0 && Number(e.target.value) <= 100) {setHistoryWindow(Number(e.target.value))}}}></input>
              <label className="form-label">
                Filter: <input type="checkbox" value={filter} onChange={(e) => setFilter(e.target.value)}></input>
              </label>
              <label className="form-label">Similarity Threshold:</label>
              <input className="form-input" type="number" min="0.0" max="1.0" step="0.05" value={similarityThreshold.toString()} onChange={(e) => { if (Number(e.target.value) >= 0.0 && Number(e.target.value) <= 1.0) {setSimilarityThreshold(Number(e.target.value))}}}></input>
              <label className="form-label">
                BM25: <input type="checkbox" value={bm25} onChange={(e) => setBM25(e.target.value)}></input>
              </label>
            </div>
            )}
          </div>
          
          <button type="submit" disabled={isLoading}>
            Send
          </button>
        </div>
      </form>
    </div>
  );
};

export default ChatInterface;