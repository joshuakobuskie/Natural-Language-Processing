require('dotenv').config(); // Load environment variables
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

// Initialize Express
const app = express();
const port = 5001;

// Middleware
app.use(cors());
app.use(express.json());

// Initialize Gemini
const genAI = new GoogleGenerativeAI(process.env.API_KEY); // Use environment variable
const model = genAI.getGenerativeModel({ model: 'gemini-2.0-flash' });

// API endpoint
app.post('/api/generate', async (req, res) => {
  const { prompt } = req.body;

  // Should never occur because the frontend stops this behavior
  if (!prompt) {
    return res.status(400).json({ error: 'Prompt is required' });
  }

  try {
    // Call Gemini API
    const result = await model.generateContent(prompt);
    const response = await result.response;
    const text = response.text();

    res.json({ response: text });

  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: 'Failed to generate response' });
  }
});

// Start server
app.listen(port, () => {
  console.log(`Node.js server running on http://localhost:${port}`);
});